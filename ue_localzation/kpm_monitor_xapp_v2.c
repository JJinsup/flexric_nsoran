/*
 * SINR Monitor xApp with Cell Coordinates - Orange 기반
 * Cell ID별 좌표 정보를 포함한 SINR 데이터 출력
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

static void log_both(const char* format, ...) {
    va_list args1, args2;
    va_start(args1, format);
    va_copy(args2, args1);
    
    vprintf(format, args1);
    
    if (log_file != NULL) {
        vfprintf(log_file, format, args2);
        fflush(log_file);
    }
    
    va_end(args1);
    va_end(args2);
}

// Signal handler
static void signal_handler(int signal) {
    (void)signal;
    monitoring_active = false;
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

// 새로운 형식으로 SINR 데이터 출력 (한 줄에 모든 정보)
static void output_sinr_data_oneline(void) {
    for (size_t i = 0; i < num_measurements; i++) {
        sinr_measurement_t* m = &measurements[i];
        
        // ✅ neighbor들을 SINR 좋은 순으로 정렬
        if (m->num_neighbors > 1) {
            qsort(m->neighbors, m->num_neighbors, 
                  sizeof(m->neighbors[0]), compare_neighbors_by_sinr);
        }
        
        // 한 줄에 모든 정보 출력
        log_both("%lu,%d,%d,%.2f", 
            m->timestamp,      // timestamp
            m->ueID,          // UE ID  
            m->servingCellID, // serving cell ID
            m->servingSINR    // serving cell SINR
        );
        
        // ✅ 상위 3개 neighbor SINR (그냥 좋은 순으로)
        for (size_t j = 0; j < 3; j++) {
            if (j < m->num_neighbors) {
                log_both(",%.2f", m->neighbors[j].neighSINR);
            } else {
                // 데이터가 없으면 0.0으로 채움
                log_both(",0.0");
            }
        }
        
        // serving cell 좌표
        log_both(",%.1f,%.1f\n",
            m->servingPos ? m->servingPos->x : 0.0, // serving cell x
            m->servingPos ? m->servingPos->y : 0.0  // serving cell y
        );
    }
    
    // 측정값 배열 초기화
    num_measurements = 0;
}

// 간소화된 KPM 측정값 로깅 함수
static void log_kpm_measurements(kpm_ind_msg_format_1_t const* msg_frm_1, uint64_t timestamp)
{
    assert(msg_frm_1->meas_info_lst_len > 0);
    
    if(msg_frm_1->meas_info_lst_len != msg_frm_1->meas_data_lst_len) {
        return;
    }

    uint64_t timestamp_ms = timestamp / 1000;

    // 먼저 serving 정보를 수집
    for(size_t i = 0; i < msg_frm_1->meas_info_lst_len; i++) {
        meas_type_t const meas_type = msg_frm_1->meas_info_lst[i].meas_type;
        meas_data_lst_t const data_item = msg_frm_1->meas_data_lst[i];
        
        if(meas_type.type == NAME_MEAS_TYPE) {
            // Serving SINR 처리
            if(isMeasNameContains((char*)meas_type.name.buf, "L3servingSINR3gpp_cell_")) {
                struct InfoObj info = parseServingMsg((char*)meas_type.name.buf);
                
                if(info.cellID != UINT16_MAX && info.ueID != UINT16_MAX && 
                   data_item.meas_record_len > 0) {
                    
                    meas_record_lst_t const record_item = data_item.meas_record_lst[0];
                    
                    if(record_item.value == REAL_MEAS_VALUE || record_item.value == INTEGER_MEAS_VALUE) {
                        // 새로운 측정값 추가
                        if (num_measurements < 100) {
                            sinr_measurement_t* m = &measurements[num_measurements];
                            m->timestamp = timestamp_ms;
                            m->ueID = info.ueID;
                            m->servingCellID = info.cellID;
                            m->servingSINR = (record_item.value == REAL_MEAS_VALUE) ? 
                                           record_item.real_val : (double)record_item.int_val;
                            m->servingPos = get_cell_position(info.cellID);
                            m->num_neighbors = 0;
                            
                            num_measurements++;
                        }
                    }
                }
            }
        }
    }
    
    // 다음으로 neighbor 정보를 수집하여 기존 측정값에 추가
    for(size_t i = 0; i < msg_frm_1->meas_info_lst_len; i++) {
        meas_type_t const meas_type = msg_frm_1->meas_info_lst[i].meas_type;
        meas_data_lst_t const data_item = msg_frm_1->meas_data_lst[i];
        
        if(meas_type.type == NAME_MEAS_TYPE) {
            // Neighbor SINR 처리
            if(isMeasNameContains((char*)meas_type.name.buf, "L3neighSINRListOf_UEID_")) {
                struct InfoObj info = parseNeighMsg((char*)meas_type.name.buf);
                
                if(info.cellID != UINT16_MAX && info.ueID != UINT16_MAX) {
                    // 해당 UE의 측정값 찾기
                    for (size_t m_idx = 0; m_idx < num_measurements; m_idx++) {
                        if (measurements[m_idx].ueID == info.ueID) {
                            // neighbor 데이터가 2개씩 온다 (SINR, NeighborID)
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
        
        // 수집된 데이터를 새로운 형식(한 줄)으로 출력
        output_sinr_data_oneline();
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
