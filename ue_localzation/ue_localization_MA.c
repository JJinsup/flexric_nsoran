/*
 * SINR Monitor xApp with Cell Coordinates - Orange 기반
 * Cell ID별 좌표 정보를 포함한 SINR 데이터 출력
 * 5초간 이동평균 처리 후 전송
 * Format: timestamp, UE, serving cell ID, serving cell SINR, top 3 neighbor SINR, serving cell x, serving cell y
 */

#include "../../../../src/xApp/e42_xapp_api.h"
#include "../../../../src/util/time_now_us.h"
#include "../../../../src/util/alg_ds/ds/lock_guard/lock_guard.h"
#include "../../../../src/util/alg_ds/alg/defer.h"
#include "../../../../src/xApp/sm_ran_function.h"
#include "../../../../src/util/e.h"
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <stdarg.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>

// 전역 변수
static int socket_fd = -1;
static bool socket_connected = false;
static const char* SOCKET_PATH = "/tmp/sinr_localization.sock";
#define MAX_UE_COUNT 100
static size_t active_ue_count = 0;

// Cell Position Structure
typedef struct {
    uint16_t cellID;
    double x;
    double y;
} cell_position_t;

// Cell Position Mapping (ns-O-RAN 시뮬레이터 기준)
static cell_position_t cell_positions[] = {
    {2, 800.0, 800.0},         // gNB 1 중앙 위치 (LTE eNB + mmWave gNB 공존)
    {3, 1200.0, 800.0},        // gNB 2 동쪽 (0도, 400m)
    {4, 1000.0, 1146.0},      // gNB 3 북동쪽 (60도, 400m)
    {5, 600.0, 1146.0},       // gNB 4 북서쪽 (120도, 400m)
    {6, 400.0, 800.0},         // gNB 5 서쪽 (180도, 400m)
    {7, 600.0, 453.0},        // gNB 6 남서쪽 (240도, 400m)
    {8, 1000.0, 453.0},       // gNB 7 남동쪽 (300도, 400m)
};

static size_t num_cells = sizeof(cell_positions) / sizeof(cell_position_t);

// Global variables
static pthread_mutex_t mtx;
static bool monitoring_active = true;
static uint64_t const period_ms = 100;  
static int indication_counter = 0;
static FILE *log_file = NULL;
static bool log_to_file = true;

// Cell position 조회 함수
static cell_position_t* get_cell_position(uint16_t cellID) {
    for (size_t i = 0; i < num_cells; i++) {
        if (cell_positions[i].cellID == cellID) {
            return &cell_positions[i];
        }
    }
    return NULL; // Cell not found
}

// 기존 sinr_measurement_t 구조체 뒤에 추가
typedef struct {
    uint64_t timestamp;
    uint16_t ueID;
    uint16_t servingCellID;
    double servingSINR;
    double neighborSINR[3];
    double servingCellX, servingCellY;
} measurement_point_t;

typedef struct {
    uint16_t ue_id;
    uint16_t current_serving_cell;
    uint64_t window_start_time;
    bool window_active;
    
    measurement_point_t buffer[200];  // 5초간 최대 데이터 수
    size_t count;
} ue_adaptive_window_t;

// 함수 선언
static bool init_unix_socket(void);
static void close_unix_socket(void);
static void send_to_python(const char* data);
static ue_adaptive_window_t ue_windows[MAX_UE_COUNT];
static void log_both(const char* format, ...);  // ← 이것 추가!
static void process_measurements_to_adaptive_windows(void);


static ue_adaptive_window_t* find_ue_window(uint16_t ue_id) {
    for (size_t i = 0; i < active_ue_count; i++) {
        if (ue_windows[i].ue_id == ue_id) {
            return &ue_windows[i];
        }
    }
    return NULL;
}

static ue_adaptive_window_t* create_ue_window(uint16_t ue_id) {
    if (active_ue_count >= MAX_UE_COUNT) {
        printf("[WARNING] Max UE count reached, ignoring UE %d\n", ue_id);
        return NULL;
    }
    
    ue_adaptive_window_t* window = &ue_windows[active_ue_count++];
    memset(window, 0, sizeof(ue_adaptive_window_t));
    window->ue_id = ue_id;
    return window;
}

static void send_window_batch_to_python(ue_adaptive_window_t* window) {
    if (window->count == 0) return;
    
    // 윈도우 데이터들의 평균 계산
    double avg_serving_sinr = 0;
    double avg_neighbor_sinr[3] = {0};
    uint64_t avg_timestamp = 0;
    
    for (size_t i = 0; i < window->count; i++) {
        avg_serving_sinr += window->buffer[i].servingSINR;
        avg_timestamp += window->buffer[i].timestamp;
        for (int j = 0; j < 3; j++) {
            avg_neighbor_sinr[j] += window->buffer[i].neighborSINR[j];
        }
    }
    
    avg_serving_sinr /= window->count;
    avg_timestamp /= window->count;
    for (int j = 0; j < 3; j++) {
        avg_neighbor_sinr[j] /= window->count;
    }
    
    // 마지막 데이터의 cell 위치 정보 사용
    measurement_point_t* last = &window->buffer[window->count - 1];
    
    // CSV 형태로 전송
    char batch_line[256];
    snprintf(batch_line, sizeof(batch_line),
        "%lu,%d,%d,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f\n",
        avg_timestamp,
        window->ue_id,
        window->current_serving_cell,
        avg_serving_sinr,
        avg_neighbor_sinr[0],
        avg_neighbor_sinr[1], 
        avg_neighbor_sinr[2],
        last->servingCellX,
        last->servingCellY
    );
    
    log_both("%s", batch_line);
}

static void reset_window(ue_adaptive_window_t* window, uint64_t new_start_time) {
    window->count = 0;
    window->window_start_time = new_start_time;
    window->current_serving_cell = 0;  // 다음 데이터에서 설정됨
    window->window_active = false;
}

// Unix Domain Socket 초기화 함수
static bool init_unix_socket(void) {
    struct sockaddr_un addr;
    
    // 소켓 생성
    socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (socket_fd == -1) {
        printf("[SOCKET] Failed to create socket: %s\n", strerror(errno));
        return false;
    }
    
    // 주소 설정
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);
    
    // Python 서버에 연결 (최대 5회 재시도)
    for (int i = 0; i < 5; i++) {
        if (connect(socket_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            socket_connected = true;
            printf("[SOCKET] ✅ Connected to Python receiver at %s\n", SOCKET_PATH);
            return true;
        }
        
        if (i == 0) {
            printf("[SOCKET] ⚠️  Python receiver not ready. Retrying...\n");
            printf("[SOCKET] 💡 Start 'python3 localization.py' first!\n");
        }
        sleep(1);
    }
    
    printf("[SOCKET] ❌ Failed to connect after 5 attempts\n");
    close(socket_fd);
    socket_fd = -1;
    return false;
}

// Unix Domain Socket 종료
static void close_unix_socket(void) {
    if (socket_fd != -1) {
        close(socket_fd);
        socket_fd = -1;
        socket_connected = false;
        printf("[SOCKET] 🔌 Socket closed\n");
    }
}

// Python으로 데이터 전송
static void send_to_python(const char* data) {
    if (!socket_connected || socket_fd == -1) {
        return;
    }
    
    ssize_t bytes_sent = send(socket_fd, data, strlen(data), MSG_NOSIGNAL);
    if (bytes_sent == -1) {
        if (errno == EPIPE || errno == ECONNRESET) {
            printf("[SOCKET] ❌ Connection lost to Python receiver\n");
            socket_connected = false;
        }
    }
}

// Orange 스타일 KPM Label 생성
static label_info_lst_t fill_kpm_label(void)
{
    label_info_lst_t label_item = {0};
    label_item.noLabel = ecalloc(1, sizeof(enum_value_e));
    *label_item.noLabel = TRUE_ENUM_VALUE;
    return label_item;
}

static test_info_lst_t filter_predicate(test_cond_type_e type, test_cond_e cond, int value)
{
    test_info_lst_t dst = {0};

    dst.test_cond_type = type;
    dst.IsStat = TRUE_TEST_COND_TYPE;

    dst.test_cond = calloc(1, sizeof(test_cond_e));
    assert(dst.test_cond != NULL && "Memory allocation failed for test_cond");
    *dst.test_cond = cond;

    dst.test_cond_value = calloc(1, sizeof(test_cond_value_t));
    assert(dst.test_cond_value != NULL && "Memory allocation failed for test_cond_value");
    dst.test_cond_value->type = INTEGER_TEST_COND_VALUE;

    int64_t *int_value = calloc(1, sizeof(int64_t));
    assert(int_value != NULL && "Memory allocation failed for int_value");
    *int_value = value; 
    dst.test_cond_value->int_value = int_value;
    return dst;
}

static kpm_act_def_format_1_t fill_act_def_frm_1(ric_report_style_item_t const* report_item)
{
    assert(report_item != NULL);

    kpm_act_def_format_1_t ad_frm_1 = {0};

    size_t const sz = report_item->meas_info_for_action_lst_len;

    ad_frm_1.meas_info_lst_len = sz;
    ad_frm_1.meas_info_lst = calloc(sz, sizeof(meas_info_format_1_lst_t));
    assert(ad_frm_1.meas_info_lst != NULL && "Memory exhausted");

    for (size_t i = 0; i < sz; i++) {
        meas_info_format_1_lst_t* meas_item = &ad_frm_1.meas_info_lst[i];
        meas_item->meas_type.type = NAME_MEAS_TYPE;
        meas_item->meas_type.name = copy_byte_array(report_item->meas_info_for_action_lst[i].name);

        meas_item->label_info_lst_len = 1;
        meas_item->label_info_lst = ecalloc(1, sizeof(label_info_lst_t));
        meas_item->label_info_lst[0] = fill_kpm_label();
    }

    ad_frm_1.gran_period_ms = period_ms;
    ad_frm_1.cell_global_id = NULL;

#if defined KPM_V2_03 || defined KPM_V3_00
    ad_frm_1.meas_bin_range_info_lst_len = 0;
    ad_frm_1.meas_bin_info_lst = NULL;
#endif

    return ad_frm_1;
}

// 기존 log_both 함수를 수정 (파일 + 콘솔 + 소켓)
static void log_both(const char* format, ...) {
    va_list args1, args2, args3;
    va_start(args1, format);
    va_copy(args2, args1);
    va_copy(args3, args1);
    
    // 콘솔 출력
    vprintf(format, args1);
    
    // 파일 출력 (기존 로직)
    if (log_file != NULL) {
        vfprintf(log_file, format, args2);
        fflush(log_file);
    }
    
    // Python으로 실시간 전송 (CSV 헤더는 제외)
    if (socket_connected) {
        char buffer[512];
        vsnprintf(buffer, sizeof(buffer), format, args3);
        
        // CSV 헤더 라인은 전송하지 않음 (timestamp로 시작하는 데이터만 전송)
        if (strstr(buffer, "timestamp,UE_ID") == NULL && strlen(buffer) > 10) {
            send_to_python(buffer);
        }
    }
    
    va_end(args1);
    va_end(args2);
    va_end(args3);
}

// Signal handler
static void signal_handler(int signal) {
    (void)signal;
    printf("\n🛑 Received signal %d\n", signal);
    monitoring_active = false;
    close_unix_socket(); //  시그널 시 소켓 정리
}

// Orange 스타일 helper 함수들
static bool eq_sm(sm_ran_function_t const* elem, int const id) {
    return elem->id == id;
}

static size_t find_sm_idx(sm_ran_function_t* rf, size_t sz, 
                         bool (*f)(sm_ran_function_t const*, int const), int const id) {
    for (size_t i = 0; i < sz; i++) {
        if (f(&rf[i], id))
            return i;
    }
    assert(0 != 0 && "SM ID could not be found in the RAN Function List");
}

// Orange 스타일 측정값 파싱 구조체
struct InfoObj { 
    uint16_t cellID;
    uint16_t ueID;
};

// Orange 스타일 문자열 파싱 함수들
static struct InfoObj parseServingMsg(const char* msg) {
    struct InfoObj info;
    int ret = sscanf(msg, "L3servingSINR3gpp_cell_%hd_UEID_%hd", &info.cellID, &info.ueID);
    
    if (ret == 2)
        return info;
    
    info.cellID = -1;
    info.ueID = -1;
    return info;
}

static struct InfoObj parseNeighMsg(const char* msg) {
    struct InfoObj info;
    int ret = sscanf(msg, "L3neighSINRListOf_UEID_%hd_of_Cell_%hd", &info.ueID, &info.cellID);
    
    if (ret == 2)
        return info;
    
    info.ueID = -1;
    info.cellID = -1;
    return info;
}

static bool isMeasNameContains(const char* meas_name, const char* name) {
    return strncmp(meas_name, name, strlen(name)) == 0;
}

// SINR 데이터 구조체 (neighbor와 serving 정보 저장)
typedef struct {
    uint64_t timestamp;
    uint16_t ueID;
    uint16_t servingCellID;
    double servingSINR;
    cell_position_t* servingPos;
    
    // Neighbor 정보들
    struct {
        uint16_t neighCellID;
        double neighSINR;
    } neighbors[10]; // 최대 10개 neighbor
    size_t num_neighbors;
} sinr_measurement_t;




// 측정값을 저장할 임시 구조체 배열
static sinr_measurement_t measurements[100]; // UE별 최대 100개
static size_t num_measurements = 0;

// neighbor SINR 정렬을 위한 비교 함수
static int compare_neighbors_by_sinr(const void* a, const void* b) {
    const struct {
        uint16_t neighCellID;
        double neighSINR;
    } *na = a, *nb = b;
    
    // 내림차순 정렬 (높은 SINR이 먼저)
    if (na->neighSINR > nb->neighSINR) return -1;
    if (na->neighSINR < nb->neighSINR) return 1;
    return 0;
}

// 간소화된 KPM 측정값 로깅 함수
static void log_kpm_measurements(kpm_ind_msg_format_1_t const* msg_frm_1, uint64_t timestamp)
{
    assert(msg_frm_1->meas_info_lst_len > 0);
    
    if(msg_frm_1->meas_info_lst_len != msg_frm_1->meas_data_lst_len) {
        return;
    }

    uint64_t timestamp_ms = timestamp / 1000;

    // 임시로 measurements 구조체들 수집 (기존 방식 유지)
    // serving 정보 먼저 수집
    for(size_t i = 0; i < msg_frm_1->meas_info_lst_len; i++) {
        meas_type_t const meas_type = msg_frm_1->meas_info_lst[i].meas_type;
        meas_data_lst_t const data_item = msg_frm_1->meas_data_lst[i];
        
        if(meas_type.type == NAME_MEAS_TYPE) {
            if(isMeasNameContains((char*)meas_type.name.buf, "L3servingSINR3gpp_cell_")) {
                struct InfoObj info = parseServingMsg((char*)meas_type.name.buf);
                
                if(info.cellID != UINT16_MAX && info.ueID != UINT16_MAX && 
                   data_item.meas_record_len > 0) {
                    
                    meas_record_lst_t const record_item = data_item.meas_record_lst[0];
                    
                    if(record_item.value == REAL_MEAS_VALUE || record_item.value == INTEGER_MEAS_VALUE) {
                        // 새로운 measurement_point_t 생성
                        measurement_point_t new_point;
                        new_point.timestamp = timestamp_ms;
                        new_point.ueID = info.ueID;
                        new_point.servingCellID = info.cellID;
                        new_point.servingSINR = (record_item.value == REAL_MEAS_VALUE) ? 
                                              record_item.real_val : (double)record_item.int_val;
                        
                        // cell 위치 정보 
                        cell_position_t* pos = get_cell_position(info.cellID);
                        new_point.servingCellX = pos ? pos->x : 0.0;
                        new_point.servingCellY = pos ? pos->y : 0.0;
                        
                        // neighbor 초기화 (나중에 채워짐)
                        for (int n = 0; n < 3; n++) {
                            new_point.neighborSINR[n] = 0.0;
                        }
                        
                        // 임시 measurements 배열에 저장 (neighbor 수집용)
                        if (num_measurements < 100) {
                            measurements[num_measurements].timestamp = timestamp_ms;
                            measurements[num_measurements].ueID = info.ueID;
                            measurements[num_measurements].servingCellID = info.cellID;
                            measurements[num_measurements].servingSINR = new_point.servingSINR;
                            measurements[num_measurements].servingPos = pos;
                            measurements[num_measurements].num_neighbors = 0;
                            
                            num_measurements++;
                        }
                    }
                }
            }
        }
    }
    
    // neighbor 정보 수집 (기존 로직 유지)
    for(size_t i = 0; i < msg_frm_1->meas_info_lst_len; i++) {
        meas_type_t const meas_type = msg_frm_1->meas_info_lst[i].meas_type;
        meas_data_lst_t const data_item = msg_frm_1->meas_data_lst[i];
        
        if(meas_type.type == NAME_MEAS_TYPE) {
            if(isMeasNameContains((char*)meas_type.name.buf, "L3neighSINRListOf_UEID_")) {
                struct InfoObj info = parseNeighMsg((char*)meas_type.name.buf);
                
                if(info.cellID != UINT16_MAX && info.ueID != UINT16_MAX) {
                    // 해당 UE의 측정값 찾기
                    for (size_t m_idx = 0; m_idx < num_measurements; m_idx++) {
                        if (measurements[m_idx].ueID == info.ueID) {
                            // neighbor 데이터 수집
                            for(size_t j = 0; j + 1 < data_item.meas_record_len; j += 2) {
                                meas_record_lst_t const sinr = data_item.meas_record_lst[j];
                                meas_record_lst_t const neighID = data_item.meas_record_lst[j + 1];
                                
                                if(sinr.value == REAL_MEAS_VALUE && neighID.value == INTEGER_MEAS_VALUE) {
                                    if (measurements[m_idx].num_neighbors < 10) {
                                        size_t n_idx = measurements[m_idx].num_neighbors;
                                        measurements[m_idx].neighbors[n_idx].neighCellID = neighID.int_val;
                                        measurements[m_idx].neighbors[n_idx].neighSINR = sinr.real_val;
                                        measurements[m_idx].num_neighbors++;
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
    
    // ⚡ 새로운 부분: 수집된 데이터를 적응적 윈도우에 추가
    process_measurements_to_adaptive_windows();
}

static void add_to_adaptive_window(measurement_point_t* point) {
    ue_adaptive_window_t* window = find_ue_window(point->ueID);
    
    // UE 윈도우가 없으면 생성
    if (window == NULL) {
        window = create_ue_window(point->ueID);
        if (window == NULL) return;  // 생성 실패
    }
    
    // 첫 번째 데이터거나 serving cell이 같으면
    if (window->count == 0 || window->current_serving_cell == point->servingCellID) {
        
        // 첫 번째 데이터면 윈도우 설정
        if (window->count == 0) {
            window->current_serving_cell = point->servingCellID;
            window->window_start_time = point->timestamp;
            window->window_active = true;
        }
        
        // 윈도우에 추가
        window->buffer[window->count++] = *point;
        
        // 5초 지났으면 전송
        if (point->timestamp - window->window_start_time >= 5000) {
            send_window_batch_to_python(window);
            reset_window(window, point->timestamp);
        }
        
    } else {
        // serving cell 바뀜! 즉시 전송하고 새 윈도우 시작
        if (window->count > 0) {  // 기존 데이터가 있으면 전송
            send_window_batch_to_python(window);
        }
        
        reset_window(window, point->timestamp);
        window->current_serving_cell = point->servingCellID;
        window->window_active = true;
        window->buffer[window->count++] = *point;
    }
}

static void process_measurements_to_adaptive_windows(void) {
    for (size_t i = 0; i < num_measurements; i++) {
        sinr_measurement_t* m = &measurements[i];
        
        // neighbor들을 SINR 좋은 순으로 정렬 (기존 로직)
        if (m->num_neighbors > 1) {
            qsort(m->neighbors, m->num_neighbors, 
                  sizeof(m->neighbors[0]), compare_neighbors_by_sinr);
        }
        
        // measurement_point_t로 변환
        measurement_point_t point;
        point.timestamp = m->timestamp;
        point.ueID = m->ueID;
        point.servingCellID = m->servingCellID;
        point.servingSINR = m->servingSINR;
        
        // top 3 neighbor SINR
        for (int j = 0; j < 3; j++) {
            point.neighborSINR[j] = ((size_t)j < m->num_neighbors) ? m->neighbors[j].neighSINR : 0.0;
        }
        
        point.servingCellX = m->servingPos ? m->servingPos->x : 0.0;
        point.servingCellY = m->servingPos ? m->servingPos->y : 0.0;
        
        // 적응적 윈도우에 추가
        add_to_adaptive_window(&point);
    }
    
    // 측정값 배열 초기화
    num_measurements = 0;
}

// Orange 스타일 메인 콜백 함수 (새로운 출력 형식)
static void sm_cb_kpm(sm_ag_if_rd_t const* rd)
{
    assert(rd != NULL);
    assert(rd->type == INDICATION_MSG_AGENT_IF_ANS_V0);
    assert(rd->ind.type == KPM_STATS_V3_0);

    kpm_ind_data_t const* ind = &rd->ind.kpm.ind;
    kpm_ric_ind_hdr_format_1_t const* hdr_frm_1 = &ind->hdr.kpm_ric_ind_hdr_format_1;
    kpm_ind_msg_format_3_t const* msg_frm_3 = &ind->msg.frm_3;

    {
        lock_guard(&mtx);
        
        // CSV 헤더 출력 (첫 번째 indication에서만)
        if (indication_counter == 0) {
            log_both("timestamp,UE_ID,serving_cell_ID,serving_cell_SINR,neighbor_1_SINR,neighbor_2_SINR,neighbor_3_SINR,serving_cell_x,serving_cell_y\n");
        }
        
        indication_counter++;

        // UE별 측정값 처리
        for (size_t i = 0; i < msg_frm_3->ue_meas_report_lst_len; i++) {
            log_kpm_measurements(&msg_frm_3->meas_report_per_ue[i].ind_msg_format_1, 
                                hdr_frm_1->collectStartTime);
        }
        
    }
}

// Orange 스타일 KPM subscription 생성
static kpm_sub_data_t gen_kpm_subs(kpm_ran_function_def_t const* ran_func)
{
    assert(ran_func != NULL);
    assert(ran_func->ric_event_trigger_style_list != NULL);

    kpm_sub_data_t kpm_sub = {0};

    assert(ran_func->ric_event_trigger_style_list[0].format_type == FORMAT_1_RIC_EVENT_TRIGGER);
    kpm_sub.ev_trg_def.type = FORMAT_1_RIC_EVENT_TRIGGER;
    kpm_sub.ev_trg_def.kpm_ric_event_trigger_format_1.report_period_ms = period_ms;

    kpm_sub.sz_ad = 1;
    kpm_sub.ad = calloc(kpm_sub.sz_ad, sizeof(kpm_act_def_t));
    assert(kpm_sub.ad != NULL && "Memory exhausted");

    ric_report_style_item_t* const report_item = &ran_func->ric_report_style_list[0];
    
    if(report_item->act_def_format_type == FORMAT_4_ACTION_DEFINITION) {
        kpm_sub.ad[0].type = FORMAT_4_ACTION_DEFINITION;
        
        kpm_sub.ad[0].frm_4.matching_cond_lst_len = 1;
        kpm_sub.ad[0].frm_4.matching_cond_lst = calloc(1, sizeof(matching_condition_format_4_lst_t));
        
        test_cond_type_e const type = IsStat_TEST_COND_TYPE;
        test_cond_e const condition = GREATERTHAN_TEST_COND;
        int const value = 2; // neighbor cell 3개 이상인 경우에만
        kpm_sub.ad[0].frm_4.matching_cond_lst[0].test_info_lst = 
            filter_predicate(type, condition, value);
        
        kpm_sub.ad[0].frm_4.action_def_format_1 = fill_act_def_frm_1(report_item);
    }

    return kpm_sub;
}

int main(int argc, char *argv[]) 
{
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // CSV 형식으로 로그 파일 열기
    log_file = fopen("sinr_ml_dataset.csv", "w");
    if (log_file == NULL) {
        log_to_file = false;
    }

    // 🔥 Unix Socket 초기화 (중요!)
    printf("[INIT] Connecting to Python receiver...\n");
    if (init_unix_socket()) {
        printf("[INIT] ✅ Python integration enabled\n");
    } else {
        printf("[INIT] ⚠️  Running without Python integration\n");
        printf("[INIT] 💡 To enable: run 'python3 localization.py' first\n");
    }

    fr_args_t args = init_fr_args(argc, argv);
    init_xapp_api(&args);
    sleep(1);
    
    e2_node_arr_xapp_t nodes = e2_nodes_xapp_api();
    defer({ free_e2_node_arr_xapp(&nodes); });
    assert(nodes.len > 0);

    pthread_mutexattr_t attr = {0};
    int rc = pthread_mutex_init(&mtx, &attr);
    assert(rc == 0);

    sm_ans_xapp_t* hndl = calloc(nodes.len, sizeof(sm_ans_xapp_t));
    assert(hndl != NULL);

    int const KPM_ran_function = 2;
    for (size_t i = 0; i < nodes.len; ++i) {
        e2_node_connected_xapp_t* n = &nodes.n[i];
        size_t const idx = find_sm_idx(n->rf, n->len_rf, eq_sm, KPM_ran_function);
        
        if (idx < n->len_rf && 
            n->rf[idx].defn.type == KPM_RAN_FUNC_DEF_E &&
            n->rf[idx].defn.kpm.ric_report_style_list != NULL) {
            
            kpm_sub_data_t kpm_sub = gen_kpm_subs(&n->rf[idx].defn.kpm);
            hndl[i] = report_sm_xapp_api(&n->id, KPM_ran_function, &kpm_sub, sm_cb_kpm);
            assert(hndl[i].success == true);
            free_kpm_sub_data(&kpm_sub);
        }
    }

    // 메인 루프
    while(monitoring_active) {
        usleep(100000);
    }

    // cleanup
    printf("\n🛑 Shutting down...\n");
    close_unix_socket(); // 🔥 중요: 소켓 종료
    if (log_file != NULL) {
        fclose(log_file);
    }

    for (int i = 0; i < nodes.len; ++i) {
        if (hndl[i].success == true)
            rm_report_sm_xapp_api(hndl[i].u.handle);
    }
    free(hndl);

    while(try_stop_xapp_api() == false)
        usleep(1000);

    rc = pthread_mutex_destroy(&mtx);
    assert(rc == 0);

    return 0;
}
