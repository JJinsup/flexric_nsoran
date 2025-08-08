/*
 * SINR Monitor xApp with 5-second Moving Average (Simulation Time)
 * 🔥 시뮬레이션 시간 기준 5초 간격 이동평균 처리
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
#include <math.h>
// =============================================================================
// CONSTANTS & GLOBAL VARIABLES
// =============================================================================
#define WINDOW_SIZE 1
#define NUM_CELLS 7
#define MIN_NEIGHBORS_REQUIRED 3 // 최소 필요 neighbor 개수
#define TOTAL_UES 28                             // 전체 UE 개수

static int socket_fd = -1;
static bool socket_connected = false;
static const char* SOCKET_PATH = "/tmp/sinr_localization.sock";
static pthread_mutex_t mtx;
static bool monitoring_active = true;
static uint64_t const period_ms = 100;  
static int indication_counter = 0;
static FILE *log_file = NULL;
static uint64_t current_sequence_timestamp = 0;  // 0, 1,2
static int current_burst_ue_count = 0;           // 현재 burst에서 받은 UE 개수
static bool burst_sequence_assigned[TOTAL_UES] = {false}; // UE별 sequence 할당 여부
// =============================================================================
// DATA STRUCTURES
// =============================================================================

// Cell Position Structure
typedef struct {
    uint16_t cellID;
    int x;
    int y;
} cell_position_t;

// Cell Position Mapping (ns-O-RAN 시뮬레이터 기준)
static cell_position_t cell_positions[] = {
    {2, 800, 800},         // gNB 1 중앙 위치
    {3, 1300, 800},        // gNB 2 동쪽
    {4, 1050, 1233},       // gNB 3 북동쪽
    {5, 550, 1233},        // gNB 4 북서쪽
    {6, 300, 800},         // gNB 5 서쪽
    {7, 550, 366},         // gNB 6 남서쪽
    {8, 1050, 366},        // gNB 7 남동쪽
};

// 🔥 이동평균을 위한 UE 데이터 버퍼
typedef struct {
    uint16_t ueID;
    uint16_t servingCellID;
    
    // Circular buffer for serving SINR
    double serving_sinr_buffer[WINDOW_SIZE];
    int serving_buffer_idx;
    int serving_sample_count;
    
    // Neighbor buffers
    struct {
        uint16_t neighCellID;
        double sinr_buffer[WINDOW_SIZE];
        int buffer_idx;
        int sample_count;
        bool is_active;
    } neighbors[10];

    // 추가: 전체 측정값 히스토리 (이동평균용)
    struct {
        double serving_sinr;
        double neighbor_sinrs[10];
        uint16_t neighbor_ids[10];
        int active_neighbor_count;
        uint64_t timestamp;
    } measurement_history[WINDOW_SIZE];

    int active_neighbors;
    int history_idx;        // 현재 쓰기 위치
    int history_count;      // 누적된 측정값 개수
    uint64_t last_timestamp;
} ue_buffer_t;

// UE 버퍼 (최대 20개 UE)
static ue_buffer_t ue_buffers[28];
static int num_active_ues = 0;

// Orange 스타일 측정값 파싱 구조체
struct InfoObj { 
    uint16_t cellID;
    uint16_t ueID;
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================
// UE별 sequence timestamp 저장
static uint64_t ue_sequence_timestamps[TOTAL_UES + 1] = {0}; // UE ID는 1~28

static uint64_t assign_sequence_timestamp(uint16_t ue_id) {
    // 🔥 이미 현재 burst에서 sequence가 할당된 UE면 기존 값 반환
    if (burst_sequence_assigned[ue_id]) {
        return ue_sequence_timestamps[ue_id];
    }
    
    // 🔥 새로운 UE면 현재 sequence timestamp 할당
    ue_sequence_timestamps[ue_id] = current_sequence_timestamp;
    burst_sequence_assigned[ue_id] = true;
    current_burst_ue_count++;
    
    printf("📊 UE_%d assigned sequence: %lu ms (count: %d/%d)\n", 
           ue_id, current_sequence_timestamp, current_burst_ue_count, TOTAL_UES);
    
    // 🔥 28개 UE가 모두 들어오면 다음 sequence로 준비
    if (current_burst_ue_count >= TOTAL_UES) {
        printf("✅ Burst complete! Moving to next sequence: %lu → %lu\n", 
               current_sequence_timestamp, current_sequence_timestamp + 1);
        
        // 다음 burst 준비
        current_sequence_timestamp += 1;
        current_burst_ue_count = 0;
        
        // 모든 UE의 할당 상태 초기화
        for (int i = 0; i <= TOTAL_UES; i++) {
            burst_sequence_assigned[i] = false;
        }
    }
    
    return ue_sequence_timestamps[ue_id];
}

// Cell position 조회
static cell_position_t* get_cell_position(uint16_t cellID) {
    for (size_t i = 0; i < NUM_CELLS; i++) {
        if (cell_positions[i].cellID == cellID) {
            return &cell_positions[i];
        }
    }
    return NULL;
}

// UE 버퍼 찾기 또는 생성
static ue_buffer_t* get_or_create_ue_buffer(uint16_t ueID) {
    // 기존 UE 찾기
    for (int i = 0; i < num_active_ues; i++) {
        if (ue_buffers[i].ueID == ueID) {
            return &ue_buffers[i];
        }
    }
    
    // 새로운 UE 생성
    if (num_active_ues < 28) {
        ue_buffer_t* new_ue = &ue_buffers[num_active_ues];
        memset(new_ue, 0, sizeof(ue_buffer_t));
        new_ue->ueID = ueID;
        num_active_ues++;
        printf("📱 New UE buffer created: UE_%d (total: %d)\n", ueID, num_active_ues);
        return new_ue;
    }
    
    return NULL;  // 버퍼 가득참
}

// Orange 스타일 문자열 파싱 함수들
static struct InfoObj parseServingMsg(const char* msg) {
    struct InfoObj info;
    int ret = sscanf(msg, "L3servingSINR3gpp_cell_%hd_UEID_%hd", &info.cellID, &info.ueID);
    
    if (ret == 2) return info;
    
    info.cellID = -1;
    info.ueID = -1;
    return info;
}

static struct InfoObj parseNeighMsg(const char* msg) {
    struct InfoObj info;
    int ret = sscanf(msg, "L3neighSINRListOf_UEID_%hd_of_Cell_%hd", &info.ueID, &info.cellID);
    
    if (ret == 2) return info;
    
    info.ueID = -1;
    info.cellID = -1;
    return info;
}

static bool isMeasNameContains(const char* meas_name, const char* name) {
    return strncmp(meas_name, name, strlen(name)) == 0;
}


// =============================================================================
// 🔥 샘플 처리 함수들
// =============================================================================
// ⚡ 현재 serving SINR 바로 사용 (이동평균 없음)
static void check_and_send_ue_data(ue_buffer_t* ue_buf, uint64_t sequence_timestamp) {    
    // ⚡ 현재 serving SINR 바로 사용
    int current_idx = (ue_buf->history_idx - 1 + WINDOW_SIZE) % WINDOW_SIZE;
    double serving_sinr_ma = ue_buf->measurement_history[current_idx].serving_sinr;
    
    if (isnan(serving_sinr_ma)) {
        return; // 조용히 스킵
    }
    
    // ⚡ Neighbor 데이터 수집 및 정렬
    typedef struct {
        uint16_t cellID;
        double sinr;
        int x, y;
    } neighbor_data_t;
    
    neighbor_data_t neighbors[10];
    int neighbor_count = 0;
    int active_neighbor_count = ue_buf->measurement_history[current_idx].active_neighbor_count;
    
    for (int i = 0; i < active_neighbor_count && neighbor_count < 10; i++) {
        uint16_t neighID = ue_buf->measurement_history[current_idx].neighbor_ids[i];
        double neighSINR = ue_buf->measurement_history[current_idx].neighbor_sinrs[i];
        
        if (neighID == ue_buf->servingCellID || isnan(neighSINR) || neighID == 0) {
            continue;
        }
        
        neighbors[neighbor_count].cellID = neighID;
        neighbors[neighbor_count].sinr = neighSINR;
        
        cell_position_t* neigh_pos = get_cell_position(neighID);
        neighbors[neighbor_count].x = neigh_pos ? neigh_pos->x : 0;
        neighbors[neighbor_count].y = neigh_pos ? neigh_pos->y : 0;
        neighbor_count++;
    }
    
    // SINR 기준 정렬 (내림차순)
    for (int i = 0; i < neighbor_count - 1; i++) {
        for (int j = i + 1; j < neighbor_count; j++) {
            if (neighbors[i].sinr < neighbors[j].sinr) {
                neighbor_data_t temp = neighbors[i];
                neighbors[i] = neighbors[j];
                neighbors[j] = temp;
            }
        }
    }
    
    // Trilateration 최소 조건 체크
    if (neighbor_count < MIN_NEIGHBORS_REQUIRED) {
        return; // 조용히 스킵
    }
    
    // Top 3 선택
    double top3_sinr[3] = {0, 0, 0};
    uint16_t top3_cell_ids[3] = {0, 0, 0};
    int top3_x[3] = {0, 0, 0};
    int top3_y[3] = {0, 0, 0};
    
    int valid_count = (neighbor_count > 3) ? 3 : neighbor_count;
    for (int i = 0; i < valid_count; i++) {
        top3_sinr[i] = neighbors[i].sinr;
        top3_cell_ids[i] = neighbors[i].cellID;
        top3_x[i] = neighbors[i].x;
        top3_y[i] = neighbors[i].y;
    }
    
    // CSV 출력
    cell_position_t* serving_pos = get_cell_position(ue_buf->servingCellID);
    char line[1024];
    snprintf(line, sizeof(line),
        "%lu,%d,%d,%d,%d,%.1f,%d,%d,%d,%.1f,%d,%d,%d,%.1f,%d,%d,%d,%.1f\n",
        sequence_timestamp, ue_buf->ueID, ue_buf->servingCellID,
        serving_pos ? serving_pos->x : 0, serving_pos ? serving_pos->y : 0,
        serving_sinr_ma,
        top3_cell_ids[0], top3_x[0], top3_y[0], top3_sinr[0],
        top3_cell_ids[1], top3_x[1], top3_y[1], top3_sinr[1],
        top3_cell_ids[2], top3_x[2], top3_y[2], top3_sinr[2]
    );
    
    // 전송
    if (log_file) {
        fprintf(log_file, "%s", line);
        fflush(log_file);
    }
    
    if (socket_connected) {
        send(socket_fd, line, strlen(line), MSG_NOSIGNAL);
    }
}

// UE별 serving SINR 샘플 추가
static void add_serving_sample(ue_buffer_t* ue_buf, uint16_t cellID, double sinr, uint64_t timestamp) {
    
    ue_buf->servingCellID = cellID;
    ue_buf->last_timestamp = timestamp;  // 원본은 last_timestamp에만 저장
    
    // 🔥 4. sequence_timestamp를 measurement_history에 저장
    ue_buf->measurement_history[ue_buf->history_idx].serving_sinr = sinr;
    ue_buf->measurement_history[ue_buf->history_idx].timestamp = 0; // ✅ sequence 사용
    ue_buf->measurement_history[ue_buf->history_idx].active_neighbor_count = 0;
    
    // Circular index 업데이트
    ue_buf->history_idx = (ue_buf->history_idx + 1) % WINDOW_SIZE;
    if (ue_buf->history_count < WINDOW_SIZE) {
        ue_buf->history_count++;
    }
    
}

// UE별 neighbor SINR 샘플 추가   🔥 추가: neighbor 데이터 수집 시에도 serving cell 체크
static void add_neighbor_sample(ue_buffer_t* ue_buf, uint16_t neighCellID, double sinr) {
    // serving cell과 같은 neighbor는 무시
    if (neighCellID == ue_buf->servingCellID) {
        return;  // serving cell과 같으면 추가하지 않음
    }
    
    // 기존 로직 계속...
    int current_idx = (ue_buf->history_idx - 1 + WINDOW_SIZE) % WINDOW_SIZE;
    
    if (ue_buf->measurement_history[current_idx].active_neighbor_count < 10) {
        int n_idx = ue_buf->measurement_history[current_idx].active_neighbor_count;
        ue_buf->measurement_history[current_idx].neighbor_ids[n_idx] = neighCellID;
        ue_buf->measurement_history[current_idx].neighbor_sinrs[n_idx] = sinr;
        ue_buf->measurement_history[current_idx].active_neighbor_count++;
    }
}

// Neighbor 정렬을 위한 구조체
typedef struct {
    uint16_t cellID;
    double avg_sinr;
} neighbor_rank_t;

// =============================================================================
// SOCKET COMMUNICATION
// =============================================================================

static bool init_unix_socket(void) {
    struct sockaddr_un addr;
    
    socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (socket_fd == -1) {
        printf("[SOCKET] Failed to create socket: %s\n", strerror(errno));
        return false;
    }
    
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);
    
    for (int i = 0; i < 5; i++) {
        if (connect(socket_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            socket_connected = true;
            printf("[SOCKET] ✅ Connected to Python receiver at %s\n", SOCKET_PATH);
            return true;
        }
        
        if (i == 0) {
            printf("[SOCKET] ⚠️  Python receiver not ready. Retrying...\n");
        }
        sleep(1);
    }
    
    printf("[SOCKET] ❌ Failed to connect after 5 attempts\n");
    close(socket_fd);
    socket_fd = -1;
    return false;
}

static void close_unix_socket(void) {
    if (socket_fd != -1) {
        close(socket_fd);
        socket_fd = -1;
        socket_connected = false;
        printf("[SOCKET] 🔌 Socket closed\n");
    }
}
// =============================================================================
// MEASUREMENT PROCESSING
// =============================================================================

static void log_kpm_measurements(kpm_ind_msg_format_1_t const* msg_frm_1, uint64_t simulation_timestamp) {
    assert(msg_frm_1->meas_info_lst_len > 0);
    
    if(msg_frm_1->meas_info_lst_len != msg_frm_1->meas_data_lst_len) {
        return;
    }

    // serving 정보 수집 및 50개 샘플 체크
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
                        double sinr = (record_item.value == REAL_MEAS_VALUE) ? 
                                     record_item.real_val : (double)record_item.int_val;
                        
                        ue_buffer_t* ue_buf = get_or_create_ue_buffer(info.ueID);
                        if (ue_buf) {
                            add_serving_sample(ue_buf, info.cellID, sinr, simulation_timestamp);
                        }
                    }
                }
            }
        }
    }
    
    // neighbor 정보 수집
    for(size_t i = 0; i < msg_frm_1->meas_info_lst_len; i++) {
        meas_type_t const meas_type = msg_frm_1->meas_info_lst[i].meas_type;
        meas_data_lst_t const data_item = msg_frm_1->meas_data_lst[i];
        
        if(meas_type.type == NAME_MEAS_TYPE) {
            if(isMeasNameContains((char*)meas_type.name.buf, "L3neighSINRListOf_UEID_")) {
                struct InfoObj info = parseNeighMsg((char*)meas_type.name.buf);
                
                if(info.cellID != UINT16_MAX && info.ueID != UINT16_MAX) {
                    ue_buffer_t* ue_buf = get_or_create_ue_buffer(info.ueID);
                    if (ue_buf) {
                        // neighbor 데이터 수집
                        for(size_t j = 0; j + 1 < data_item.meas_record_len; j += 2) {
                            meas_record_lst_t const sinr = data_item.meas_record_lst[j];
                            meas_record_lst_t const neighID = data_item.meas_record_lst[j + 1];
                            
                            if(sinr.value == REAL_MEAS_VALUE && neighID.value == INTEGER_MEAS_VALUE) {
                                add_neighbor_sample(ue_buf, neighID.int_val, sinr.real_val);
                            }
                        }
                    }
                }
            }
        }
    }
    // 🔥 6. UE별 데이터 전송
    for (int i = 0; i < num_active_ues; i++) {
        ue_buffer_t* ue_buf = &ue_buffers[i];
        if (ue_buf->history_count > 0) {
            uint64_t seq_ts = assign_sequence_timestamp(ue_buf->ueID);
            check_and_send_ue_data(ue_buf, seq_ts);
        }
    }
}

// =============================================================================
// KPM RELATED FUNCTIONS
// =============================================================================

static label_info_lst_t fill_kpm_label(void) {
    label_info_lst_t label_item = {0};
    label_item.noLabel = ecalloc(1, sizeof(enum_value_e));
    *label_item.noLabel = TRUE_ENUM_VALUE;
    return label_item;
}

static test_info_lst_t filter_predicate(test_cond_type_e type, test_cond_e cond, int value) {
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

static kpm_act_def_format_1_t fill_act_def_frm_1(ric_report_style_item_t const* report_item) {
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

// =============================================================================
// CALLBACK FUNCTIONS
// =============================================================================

static void sm_cb_kpm(sm_ag_if_rd_t const* rd) {
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
            if (log_file) {
                fprintf(log_file, "relative_timestamp,imsi,L3 serving Id(m_cellId),serving_x,serving_y,L3 serving SINR 3gpp_ma,L3 neigh Id 1 (cellId),neighbor1_x,neighbor1_y,L3 neigh SINR 3gpp 1 (convertedSinr)_ma,L3 neigh Id 2 (cellId),neighbor2_x,neighbor2_y,L3 neigh SINR 3gpp 2 (convertedSinr)_ma,L3 neigh Id 3 (cellId),neighbor3_x,neighbor3_y,L3 neigh SINR 3gpp 3 (convertedSinr)_ma\n");
                fflush(log_file);
            }
        }
        
        indication_counter++;

        // 🔥 시뮬레이션 시간 사용
        uint64_t simulation_time = hdr_frm_1->collectStartTime;
        
        // UE별 측정값 처리 (버퍼에 누적)
        for (size_t i = 0; i < msg_frm_3->ue_meas_report_lst_len; i++) {
            log_kpm_measurements(&msg_frm_3->meas_report_per_ue[i].ind_msg_format_1, 
                                simulation_time);
        }
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

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

static void signal_handler(int signal) {
    (void)signal;
    printf("\n🛑 Received signal %d\n", signal);
    monitoring_active = false;
    close_unix_socket();
}

static kpm_sub_data_t gen_kpm_subs(kpm_ran_function_def_t const* ran_func) {
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
        test_cond_e const condition = GREATERTHAN_TEST_COND; // 무조건보내게설정
        int const value = 2; // 무조건보내게설정
        kpm_sub.ad[0].frm_4.matching_cond_lst[0].test_info_lst = 
            filter_predicate(type, condition, value);
        
        kpm_sub.ad[0].frm_4.action_def_format_1 = fill_act_def_frm_1(report_item);
    }

    return kpm_sub;
}

// =============================================================================
// MAIN FUNCTION
// =============================================================================

int main(int argc, char *argv[]) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // CSV 로그 파일 열기
    log_file = fopen("trilateration_data.csv", "w");
    if (log_file == NULL) {
        printf("⚠️  Failed to open log file\n");
    }

    // Unix Socket 초기화
    printf("[INIT] 🔥 Connecting to Python receiver (5-second interval mode)...\n");
    if (init_unix_socket()) {
        printf("[INIT] ✅ Python integration enabled\n");
    } else {
        printf("[INIT] ⚠️  Running without Python integration\n");
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
    close_unix_socket();
    if (log_file != NULL) {
        fclose(log_file);
    }

    return 0;
}
