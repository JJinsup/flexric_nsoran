/*
 * SINR Monitor xApp - Orange 기반
 * 100ms 주기로 RIC Indication 메시지 수신하고 SINR 값 실시간 출력
 * PRB사용량 0% 이상인경우에만 E2->xApp으로수신됨 (format 4)
 * Based on orange_energy_saving_with_CU.c
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
#include <stdarg.h>  // va_list, va_start, va_end 등을 위해



// Global variables (Orange 스타일)
static pthread_mutex_t mtx;
static bool monitoring_active = true;
static uint64_t const period_ms = 100;  // 100ms 주기
static int indication_counter = 0;
static FILE *log_file = NULL;
static bool log_to_file = true;

// Orange 스타일 KPM Label 생성
static label_info_lst_t fill_kpm_label(void)
{
    label_info_lst_t label_item = {0};
    label_item.noLabel = ecalloc(1, sizeof(enum_value_e));
    *label_item.noLabel = TRUE_ENUM_VALUE;
    return label_item;
}

static
test_info_lst_t filter_predicate(test_cond_type_e type, test_cond_e cond, int value)
{
  test_info_lst_t dst = {0};

  dst.test_cond_type = type;
    dst.IsStat = TRUE_TEST_COND_TYPE;

    // Allocate memory for test_cond and set its value
    dst.test_cond = calloc(1, sizeof(test_cond_e));
    assert(dst.test_cond != NULL && "Memory allocation failed for test_cond");
    *dst.test_cond = cond;

    // Allocate memory for test_cond_value
    dst.test_cond_value = calloc(1, sizeof(test_cond_value_t));
    assert(dst.test_cond_value != NULL && "Memory allocation failed for test_cond_value");
    dst.test_cond_value->type = INTEGER_TEST_COND_VALUE;

    // Allocate memory for int_value and set its value
    int64_t *int_value = calloc(1, sizeof(int64_t));
    assert(int_value != NULL && "Memory allocation failed for int_value");
    *int_value = value; 
    dst.test_cond_value->int_value = int_value;
  return dst;
}

static
kpm_act_def_format_1_t fill_act_def_frm_1(ric_report_style_item_t const* report_item)
{
  assert(report_item != NULL);

  kpm_act_def_format_1_t ad_frm_1 = {0};

  size_t const sz = report_item->meas_info_for_action_lst_len;

  // [1, 65535]
  ad_frm_1.meas_info_lst_len = sz;
  ad_frm_1.meas_info_lst = calloc(sz, sizeof(meas_info_format_1_lst_t));
  assert(ad_frm_1.meas_info_lst != NULL && "Memory exhausted");

  for (size_t i = 0; i < sz; i++) {
    meas_info_format_1_lst_t* meas_item = &ad_frm_1.meas_info_lst[i];
    // 8.3.9
    // Measurement Name
    meas_item->meas_type.type = NAME_MEAS_TYPE;
    meas_item->meas_type.name = copy_byte_array(report_item->meas_info_for_action_lst[i].name);

    // [1, 2147483647]
    // 8.3.11
    meas_item->label_info_lst_len = 1;
    meas_item->label_info_lst = ecalloc(1, sizeof(label_info_lst_t));
    meas_item->label_info_lst[0] = fill_kpm_label();
  }

  // 8.3.8 [0, 4294967295]
  ad_frm_1.gran_period_ms = period_ms;

  // 8.3.20 - OPTIONAL
  ad_frm_1.cell_global_id = NULL;

#if defined KPM_V2_03 || defined KPM_V3_00
  // [0, 65535]
  ad_frm_1.meas_bin_range_info_lst_len = 0;
  ad_frm_1.meas_bin_info_lst = NULL;
#endif

  return ad_frm_1;
}

static void log_both(const char* format, ...) {
    va_list args1, args2;
    va_start(args1, format);
    va_copy(args2, args1);
    
    // Console 출력
    vprintf(format, args1);
    
    // 파일 출력
    if (log_file != NULL) {
        vfprintf(log_file, format, args2);
        fflush(log_file);
    }
    
    va_end(args1);
    va_end(args2);
}

// Signal handler for graceful shutdown
static void signal_handler(int signal) {
    (void)signal;
    //log_both("\n[INFO] Stopping KPM monitor...\n");
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
    // "L3servingSINR3gpp_cell_2_UEID_13" 형식 파싱
    int ret = sscanf(msg, "L3servingSINR3gpp_cell_%hd_UEID_%hd", &info.cellID, &info.ueID);
    
    if (ret == 2)
        return info;
    
    info.cellID = -1;
    info.ueID = -1;
    return info;
}

static struct InfoObj parseNeighMsg(const char* msg) {
    struct InfoObj info;
    // "L3neighSINRListOf_UEID_13_of_Cell_2" 형식 파싱
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

typedef void (*log_meas_value)(byte_array_t name, meas_record_lst_t meas_record);

// 간소화된 KPM 측정값 로깅 함수
static void log_kpm_measurements(kpm_ind_msg_format_1_t const* msg_frm_1)
{
    assert(msg_frm_1->meas_info_lst_len > 0);
    
    if(msg_frm_1->meas_info_lst_len != msg_frm_1->meas_data_lst_len) {
        return;
    }

    // Orange 스타일 측정값 파싱 및 처리
    for(size_t i = 0; i < msg_frm_1->meas_info_lst_len; i++) {
        meas_type_t const meas_type = msg_frm_1->meas_info_lst[i].meas_type;
        meas_data_lst_t const data_item = msg_frm_1->meas_data_lst[i];
        
        if(meas_type.type == NAME_MEAS_TYPE) {
            for(size_t j = 0; j < data_item.meas_record_len;) {
                meas_record_lst_t const record_item = data_item.meas_record_lst[j];
                
                // Serving SINR 처리 (Orange 방식)
                if(isMeasNameContains((char*)meas_type.name.buf, "L3servingSINR3gpp_cell_")) {
                    struct InfoObj info = parseServingMsg((char*)meas_type.name.buf);
                    
                    if(info.cellID != UINT16_MAX && info.ueID != UINT16_MAX) {
                        if(record_item.value == REAL_MEAS_VALUE) {
                            log_both("SERVING SINR - Cell:%d UE:%d = %.2f dB\n", 
                                   info.cellID, info.ueID, record_item.real_val);
                        } else if(record_item.value == INTEGER_MEAS_VALUE) {
                            log_both("ENCODED SERVING SINR - Cell:%d UE:%d = %d\n", 
                                   info.cellID, info.ueID, record_item.int_val);
                        }
                    }
                }
                // Neighbor SINR 처리 (Orange 방식)
                else if(isMeasNameContains((char*)meas_type.name.buf, "L3neighSINRListOf_UEID_")) {
                    struct InfoObj info = parseNeighMsg((char*)meas_type.name.buf);
                    
                    if(info.cellID != UINT16_MAX && info.ueID != UINT16_MAX) {
                        // Orange에서는 neighbor 데이터가 2개씩 온다 (SINR, NeighborID)
                        if(j + 1 < data_item.meas_record_len) {
                            meas_record_lst_t const sinr = record_item;
                            meas_record_lst_t const neighID = data_item.meas_record_lst[j + 1];
                            
                            if(sinr.value == REAL_MEAS_VALUE && neighID.value == INTEGER_MEAS_VALUE) {
                                log_both("NEIGHBOR SINR - UE:%d Serving:%d Neighbor:%d = %.2f dB\n", 
                                       info.ueID, info.cellID, neighID.int_val, sinr.real_val);
                            }
                            j += 2; // 2개 데이터를 처리했으므로 2씩 증가
                            continue;
                        }
                    }
                }
                j++;
            }
        }
    }
}

// Orange 스타일 메인 콜백 함수
// 간소화된 메인 콜백 함수
static void sm_cb_kpm(sm_ag_if_rd_t const* rd)
{
    assert(rd != NULL);
    assert(rd->type == INDICATION_MSG_AGENT_IF_ANS_V0);
    assert(rd->ind.type == KPM_STATS_V3_0);

    kpm_ind_data_t const* ind = &rd->ind.kpm.ind;
    kpm_ric_ind_hdr_format_1_t const* hdr_frm_1 = &ind->hdr.kpm_ric_ind_hdr_format_1;  // Header 추가!
    kpm_ind_msg_format_3_t const* msg_frm_3 = &ind->msg.frm_3;

    {
        lock_guard(&mtx);
        // Timestamp 정보 출력
        uint64_t const now = time_now_us();
        log_both("\n=== Indication #%d ===\n", ++indication_counter);
        log_both("📅 Current time: %lu μs\n", now);
        log_both("📨 Message timestamp: %lu μs\n", hdr_frm_1->collectStartTime);
        log_both("⏱️  Latency: %ld μs\n", now - hdr_frm_1->collectStartTime);

        // UE별 측정값 처리
        for (size_t i = 0; i < msg_frm_3->ue_meas_report_lst_len; i++) {
            log_kpm_measurements(&msg_frm_3->meas_report_per_ue[i].ind_msg_format_1); // SINR 측정값만 출력
        }
    }
}


// Orange 스타일 KPM subscription 생성
static kpm_sub_data_t gen_kpm_subs(kpm_ran_function_def_t const* ran_func)
{
    assert(ran_func != NULL);
    assert(ran_func->ric_event_trigger_style_list != NULL);

    kpm_sub_data_t kpm_sub = {0};

    // Event Trigger 설정
    assert(ran_func->ric_event_trigger_style_list[0].format_type == FORMAT_1_RIC_EVENT_TRIGGER);
    kpm_sub.ev_trg_def.type = FORMAT_1_RIC_EVENT_TRIGGER;
    kpm_sub.ev_trg_def.kpm_ric_event_trigger_format_1.report_period_ms = period_ms;

    // Action Definition - RAN function의 report style 사용
    kpm_sub.sz_ad = 1;
    kpm_sub.ad = calloc(kpm_sub.sz_ad, sizeof(kpm_act_def_t));
    assert(kpm_sub.ad != NULL && "Memory exhausted");

    // Orange처럼 report style에서 가져오기
    ric_report_style_item_t* const report_item = &ran_func->ric_report_style_list[0];
    
    // Report Style 4 사용 (matching condition 포함)
    if(report_item->act_def_format_type == FORMAT_4_ACTION_DEFINITION) {
        kpm_sub.ad[0].type = FORMAT_4_ACTION_DEFINITION;
        
        // Matching condition 설정
        kpm_sub.ad[0].frm_4.matching_cond_lst_len = 1;
        kpm_sub.ad[0].frm_4.matching_cond_lst = calloc(1, sizeof(matching_condition_format_4_lst_t));
        
        // Filter 설정
        test_cond_type_e const type = IsStat_TEST_COND_TYPE;
        test_cond_e const condition = LESSTHAN_TEST_COND;
        int const value = 0;
        kpm_sub.ad[0].frm_4.matching_cond_lst[0].test_info_lst = 
            filter_predicate(type, condition, value);
        
        // Action Definition Format 1
        kpm_sub.ad[0].frm_4.action_def_format_1 = fill_act_def_frm_1(report_item);
    }

    return kpm_sub;
}

int main(int argc, char *argv[]) 
{
    // Signal handler 설정
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // 로그 파일 열기
    log_file = fopen("sinr_log.txt", "w");
    if (log_file == NULL) {
        log_to_file = false;
    }

    // xApp 초기화
    fr_args_t args = init_fr_args(argc, argv);
    init_xapp_api(&args);
    sleep(1);
    
    // 노드 연결
    e2_node_arr_xapp_t nodes = e2_nodes_xapp_api();
    defer({ free_e2_node_arr_xapp(&nodes); });
    assert(nodes.len > 0);

    // mutex 초기화
    pthread_mutexattr_t attr = {0};
    int rc = pthread_mutex_init(&mtx, &attr);
    assert(rc == 0);

    // KPM subscription
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

    // 메인 루프 (상태 메시지 제거)
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
