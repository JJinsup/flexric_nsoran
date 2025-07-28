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

// =============================================================================
// CONSTANTS & GLOBAL VARIABLES
// =============================================================================

static int socket_fd = -1;
static bool socket_connected = false;
static const char* SOCKET_PATH = "/tmp/sinr_localization.sock";
static pthread_mutex_t mtx;
static bool monitoring_active = true;
static uint64_t const period_ms = 100;  
static int indication_counter = 0;
static FILE *log_file = NULL;

// 🔥 시뮬레이션 시간 기준 5초 간격 처리
static const uint64_t PREDICTION_INTERVAL_US = 5000000;  // 5초 (마이크로초)
static uint64_t last_prediction_time = 0;

// =============================================================================
// DATA STRUCTURES
// =============================================================================

// Cell Position Structure
typedef struct {
    uint16_t cellID;
    double x;
    double y;
} cell_position_t;

// Cell Position Mapping (ns-O-RAN 시뮬레이터 기준)
static cell_position_t cell_positions[] = {
    {2, 800.0, 800.0},         // gNB 1 중앙 위치
    {3, 1200.0, 800.0},        // gNB 2 동쪽
    {4, 1000.0, 1146.0},       // gNB 3 북동쪽
    {5, 600.0, 1146.0},        // gNB 4 북서쪽
    {6, 400.0, 800.0},         // gNB 5 서쪽
    {7, 600.0, 453.0},         // gNB 6 남서쪽
    {8, 1000.0, 453.0},        // gNB 7 남동쪽
};
static size_t num_cells = sizeof(cell_positions) / sizeof(cell_position_t);

// 🔥 이동평균을 위한 UE 데이터 버퍼
typedef struct {
    uint16_t ueID;
    uint16_t servingCellID;
    
    // SINR 값들 (이동평균 계산용)
    double serving_sinr_sum;
    int serving_sinr_count;
    
    struct {
        uint16_t neighCellID;
        double sinr_sum;
        int sinr_count;
    } neighbors[10];
    int active_neighbors;
    
    bool has_data;
    uint64_t first_timestamp;
    uint64_t last_timestamp;
} ue_buffer_t;

// UE 버퍼 (최대 20개 UE)
static ue_buffer_t ue_buffers[20];
static int num_active_ues = 0;

// Orange 스타일 측정값 파싱 구조체
struct InfoObj { 
    uint16_t cellID;
    uint16_t ueID;
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

// Cell position 조회
static cell_position_t* get_cell_position(uint16_t cellID) {
    for (size_t i = 0; i < num_cells; i++) {
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
    if (num_active_ues < 20) {
        ue_buffer_t* new_ue = &ue_buffers[num_active_ues];
        memset(new_ue, 0, sizeof(ue_buffer_t));
        new_ue->ueID = ueID;
        new_ue->has_data = false;
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
// 🔥 이동평균 처리 함수들
// =============================================================================

// UE 버퍼에 serving SINR 추가
static void add_serving_sinr_to_buffer(ue_buffer_t* ue_buf, uint16_t cellID, double sinr, uint64_t timestamp) {
    ue_buf->servingCellID = cellID;
    ue_buf->serving_sinr_sum += sinr;
    ue_buf->serving_sinr_count++;
    
    if (!ue_buf->has_data) {
        ue_buf->first_timestamp = timestamp;
        ue_buf->has_data = true;
    }
    ue_buf->last_timestamp = timestamp;
}

// UE 버퍼에 neighbor SINR 추가
static void add_neighbor_sinr_to_buffer(ue_buffer_t* ue_buf, uint16_t neighCellID, double sinr) {
    // 기존 neighbor 찾기
    for (int i = 0; i < ue_buf->active_neighbors; i++) {
        if (ue_buf->neighbors[i].neighCellID == neighCellID) {
            ue_buf->neighbors[i].sinr_sum += sinr;
            ue_buf->neighbors[i].sinr_count++;
            return;
        }
    }
    
    // 새로운 neighbor 추가
    if (ue_buf->active_neighbors < 10) {
        int idx = ue_buf->active_neighbors;
        ue_buf->neighbors[idx].neighCellID = neighCellID;
        ue_buf->neighbors[idx].sinr_sum = sinr;
        ue_buf->neighbors[idx].sinr_count = 1;
        ue_buf->active_neighbors++;
    }
}

// 🔥 이동평균 계산 및 Python 전송
static void process_and_send_averaged_data(uint64_t current_simulation_time) {
    printf("\n🧮 Processing 5-second moving averages (simulation time: %lu μs)\n", current_simulation_time);
    
    int processed_ues = 0;
    
    for (int i = 0; i < num_active_ues; i++) {
        ue_buffer_t* ue_buf = &ue_buffers[i];
        
        if (!ue_buf->has_data || ue_buf->serving_sinr_count == 0) {
            continue;
        }
        
        // 이동평균 계산
        double avg_serving_sinr = ue_buf->serving_sinr_sum / ue_buf->serving_sinr_count;
        
        // Top 3 neighbor 선별 (평균 SINR 기준)
        double neighbor_sinr[3] = {0.0, 0.0, 0.0};
        uint16_t neighbor_ids[3] = {0, 0, 0};
        
        // neighbor들을 평균 SINR로 정렬
        for (int n = 0; n < ue_buf->active_neighbors && n < 3; n++) {
            if (ue_buf->neighbors[n].sinr_count > 0) {
                neighbor_sinr[n] = ue_buf->neighbors[n].sinr_sum / ue_buf->neighbors[n].sinr_count;
                neighbor_ids[n] = ue_buf->neighbors[n].neighCellID;
            }
        }
        
        // Cell 위치 정보
        cell_position_t* serving_pos = get_cell_position(ue_buf->servingCellID);
        
        // CSV 형태로 출력 및 Python 전송
        char line[256];
        snprintf(line, sizeof(line),
            "%lu,%d,%d,%.2f,%d,%.2f,%d,%.2f,%d,%.2f,%.1f,%.1f\n",
            current_simulation_time / 1000,  // 밀리초로 변환
            ue_buf->ueID,
            ue_buf->servingCellID,
            avg_serving_sinr,
            neighbor_ids[0], neighbor_sinr[0],
            neighbor_ids[1], neighbor_sinr[1],
            neighbor_ids[2], neighbor_sinr[2],
            serving_pos ? serving_pos->x : 0.0,
            serving_pos ? serving_pos->y : 0.0
        );
        
        // 파일 및 소켓 전송
        if (log_file) {
            fprintf(log_file, "%s", line);
            fflush(log_file);
        }
        
        if (socket_connected) {
            send(socket_fd, line, strlen(line), MSG_NOSIGNAL);
        }
        
        printf("📊 UE_%d: Serving=%.1f dB (Cell %d), Neighbors=[%.1f, %.1f, %.1f] dB, Samples=%d\n",
               ue_buf->ueID, avg_serving_sinr, ue_buf->servingCellID,
               neighbor_sinr[0], neighbor_sinr[1], neighbor_sinr[2],
               ue_buf->serving_sinr_count);
        
        processed_ues++;
    }
    
    printf("✅ Processed %d UEs with 5-second moving averages\n", processed_ues);
    
    // 🔥 버퍼 초기화 (다음 5초 구간을 위해)
    for (int i = 0; i < num_active_ues; i++) {
        ue_buffer_t* ue_buf = &ue_buffers[i];
        ue_buf->serving_sinr_sum = 0.0;
        ue_buf->serving_sinr_count = 0;
        
        for (int n = 0; n < ue_buf->active_neighbors; n++) {
            ue_buf->neighbors[n].sinr_sum = 0.0;
            ue_buf->neighbors[n].sinr_count = 0;
        }
        ue_buf->active_neighbors = 0;
        ue_buf->has_data = false;
    }
}

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

    // 🔥 시뮬레이션 시간 기준 5초 체크
    if (last_prediction_time == 0) {
        last_prediction_time = simulation_timestamp;
        printf("🕐 First prediction timestamp: %lu μs\n", simulation_timestamp);
    }
    
    bool should_process = (simulation_timestamp - last_prediction_time) >= PREDICTION_INTERVAL_US;

    // serving 정보 수집 (버퍼에 누적)
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
                        
                        // 🔥 UE 버퍼에 추가
                        ue_buffer_t* ue_buf = get_or_create_ue_buffer(info.ueID);
                        if (ue_buf) {
                            add_serving_sinr_to_buffer(ue_buf, info.cellID, sinr, simulation_timestamp);
                        }
                    }
                }
            }
        }
    }
    
    // neighbor 정보 수집 (버퍼에 누적)
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
                                add_neighbor_sinr_to_buffer(ue_buf, neighID.int_val, sinr.real_val);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // 🔥 5초가 지났으면 이동평균 계산 및 전송
    if (should_process) {
        process_and_send_averaged_data(simulation_timestamp);
        last_prediction_time = simulation_timestamp;
        printf("⏰ Next prediction scheduled at: %lu μs\n", last_prediction_time + PREDICTION_INTERVAL_US);
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
                fprintf(log_file, "timestamp,UE_ID,serving_cell_ID,serving_cell_SINR,neighbor1_ID,neighbor_1_SINR,neighbor2_ID,neighbor_2_SINR,neighbor3_ID,neighbor_3_SINR,serving_cell_x,serving_cell_y\n");
                fflush(log_file);
            }
            printf("📋 CSV header written\n");
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
    log_file = fopen("sinr_5sec_moving_average.csv", "w");
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
