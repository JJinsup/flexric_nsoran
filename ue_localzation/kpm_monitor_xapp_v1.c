/*
 * KPM Monitor xApp - Orange 기반
 * 100ms 주기로 RIC Indication 메시지 수신하고 SINR 값 실시간 출력
 * 인코딩된 SINR < 40 인경우에만 E2->xApp으로수신됨
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

// Orange 스타일 측정값 출력 함수들
static void log_real_value(byte_array_t name, meas_record_lst_t meas_record)
{
    if (cmp_str_ba("DRB.RlcSduDelayDl", name) == 0) {
        //log_both("📊 DRB.RlcSduDelayDl = %.2f [μs]\n", meas_record.real_val);
    } else if (cmp_str_ba("DRB.UEThpDl", name) == 0) {
        //log_both("📈 DRB.UEThpDl = %.2f [kbps]\n", meas_record.real_val);
    } else if (cmp_str_ba("DRB.UEThpUl", name) == 0) {
        //log_both("📉 DRB.UEThpUl = %.2f [kbps]\n", meas_record.real_val);
    } else if (strncmp((char*)name.buf, "L3servingSINR3gpp_cell_", strlen("L3servingSINR3gpp_cell_")) == 0) {
        log_both("🔥 SERVING SINR: %s = %.4f [dB]\n", (char*)name.buf, meas_record.real_val);
    } else if (strncmp((char*)name.buf, "L3neighSINRListOf_UEID_", strlen("L3neighSINRListOf_UEID_")) == 0) {
        log_both("📡 NEIGHBOR SINR: %s = %.4f [dB]\n", (char*)name.buf, meas_record.real_val);
    } else {
        //log_both("📋 OTHER REAL: %s = %.6f\n", (char*)name.buf, meas_record.real_val);
    }
}

static void log_int_value(byte_array_t name, meas_record_lst_t meas_record)
{
    if (cmp_str_ba("RRU.PrbTotDl", name) == 0) {
        //log_both("📶 RRU.PrbTotDl = %d [PRBs]\n", meas_record.int_val);
    } else if (cmp_str_ba("RRU.PrbTotUl", name) == 0) {
        //log_both("📶 RRU.PrbTotUl = %d [PRBs]\n", meas_record.int_val);
    } else if (cmp_str_ba("DRB.PdcpSduVolumeDL", name) == 0) {
        //log_both("💾 DRB.PdcpSduVolumeDL = %d [kb]\n", meas_record.int_val);
    } else if (cmp_str_ba("DRB.PdcpSduVolumeUL", name) == 0) {
        //log_both("💾 DRB.PdcpSduVolumeUL = %d [kb]\n", meas_record.int_val);
    } else if (strncmp((char*)name.buf, "L3servingSINR3gpp", strlen("L3servingSINR3gpp")) == 0) {
        log_both("🎯 ENCODED SERVING SINR: %s = %d (encoded)\n", (char*)name.buf, meas_record.int_val);
    } else if (strncmp((char*)name.buf, "L3neighSINR", strlen("L3neighSINR")) == 0) {
        log_both("🔀 ENCODED NEIGHBOR SINR: %s = %d (encoded)\n", (char*)name.buf, meas_record.int_val);
    } else {
        //log_both("📊 OTHER INT: %s = %d\n", (char*)name.buf, meas_record.int_val);
    }
}

typedef void (*log_meas_value)(byte_array_t name, meas_record_lst_t meas_record);

static log_meas_value get_meas_value[END_MEAS_VALUE] = {
    log_int_value,
    log_real_value,
    NULL,
};

static void match_meas_name_type(meas_type_t meas_type, meas_record_lst_t meas_record)
{
    // Get the value of the Measurement
    get_meas_value[meas_record.value](meas_type.name, meas_record);
}

static void match_id_meas_type(meas_type_t meas_type, meas_record_lst_t meas_record)
{
    (void)meas_type;
    (void)meas_record;
    assert(false && "ID Measurement Type not yet supported");
}

typedef void (*check_meas_type)(meas_type_t meas_type, meas_record_lst_t meas_record);

static check_meas_type match_meas_type[END_MEAS_TYPE] = {
    match_meas_name_type,
    match_id_meas_type,
};

// Orange 스타일 KPM 측정값 로깅 함수
static void log_kpm_measurements(kpm_ind_msg_format_1_t const* msg_frm_1)
{
    assert(msg_frm_1->meas_info_lst_len > 0 && "Cannot correctly print measurements");
    
    if(msg_frm_1->meas_info_lst_len != msg_frm_1->meas_data_lst_len) {
        return;
    }

    // 현재 UE의 정보를 저장할 변수들
    int current_ue = -1;
    int current_cell = -1;
    double serving_sinr = 0.0;
    
    // Neighbor 정보를 임시 저장할 구조체
    struct {
        int neighbor_id;
        double sinr;
    } neighbors[20];
    int neighbor_count = 0;

    // Orange 스타일 측정값 처리
    for(size_t i = 0; i < msg_frm_1->meas_info_lst_len; i++) {
        meas_type_t const meas_type = msg_frm_1->meas_info_lst[i].meas_type;
        meas_data_lst_t const data_item = msg_frm_1->meas_data_lst[i];
        
        for(size_t j = 0; j < data_item.meas_record_len;) {
            meas_record_lst_t const record_item = data_item.meas_record_lst[j];
            
            if(meas_type.type == NAME_MEAS_TYPE) {
                // Serving SINR 파싱
                if(isMeasNameContains((char*)meas_type.name.buf, "L3servingSINR3gpp_cell_")) {
                    // 이전 UE 정보가 있으면 출력
                    if(current_ue != -1 && neighbor_count > 0) {
                        log_both("╰─ Neighbors: ");
                        for(int k = 0; k < neighbor_count; k++) {
                            log_both("Cell %d (%.2f dB)%s", 
                                   neighbors[k].neighbor_id, 
                                   neighbors[k].sinr,
                                   k < neighbor_count-1 ? ", " : "\n");
                        }
                        neighbor_count = 0;
                    }
                    
                    struct InfoObj info = parseServingMsg((char*)meas_type.name.buf);
                    serving_sinr = record_item.real_val;
                    current_ue = info.ueID;
                    current_cell = info.cellID;
                    
                    log_both("\n📱 UE %d - Cell %d: %.2f dB\n", 
                           info.ueID, info.cellID, serving_sinr);
                           
                } else if(isMeasNameContains((char*)meas_type.name.buf, "L3neighSINRListOf_UEID_")) {
                    // Neighbor SINR 파싱
                    struct InfoObj info = parseNeighMsg((char*)meas_type.name.buf);
                    
                    if(info.ueID == current_ue && neighbor_count < 20) {
                        meas_record_lst_t const sinr = record_item;
                        meas_record_lst_t const NeighbourID = data_item.meas_record_lst[j + 1];
                        
                        neighbors[neighbor_count].neighbor_id = NeighbourID.int_val;
                        neighbors[neighbor_count].sinr = sinr.real_val;
                        neighbor_count++;
                    }
                    j += 2;
                    continue;
                }
            }
            j++;
        }
    }
    
    // 마지막 UE의 neighbor 정보 출력
    if(current_ue != -1 && neighbor_count > 0) {
        log_both("╰─ Neighbors: ");
        for(int k = 0; k < neighbor_count; k++) {
            log_both("Cell %d (%.2f dB)%s", 
                   neighbors[k].neighbor_id, 
                   neighbors[k].sinr,
                   k < neighbor_count-1 ? ", " : "\n");
        }
    }
}

// Orange 스타일 UE ID 로깅
static void log_ue_id_e2sm(ue_id_e2sm_t const ue_id_e2sm) {
    switch (ue_id_e2sm.type) {
        case GNB_UE_ID_E2SM:
            if (ue_id_e2sm.gnb.ran_ue_id != NULL)
                log_both("👤 UE ID (GNB): 0x%lx\n", *ue_id_e2sm.gnb.ran_ue_id);
            else
                log_both("👤 UE ID (GNB): ran_ue_id is NULL\n");
                
            if (ue_id_e2sm.gnb.gnb_cu_ue_f1ap_lst != NULL)
                //log_both("   F1AP ID Count: %zu\n", ue_id_e2sm.gnb.gnb_cu_ue_f1ap_lst_len);
            break;
            
        case GNB_DU_UE_ID_E2SM:
            //log_both("👤 UE ID (GNB_DU): %u\n", ue_id_e2sm.gnb_du.gnb_cu_ue_f1ap);
            break;
            
        case GNB_CU_UP_UE_ID_E2SM:
            //log_both("👤 UE ID (GNB_CU_UP): %u\n", ue_id_e2sm.gnb_cu_up.gnb_cu_cp_ue_e1ap);
            break;
            
        case NG_ENB_UE_ID_E2SM:
            //log_both("👤 UE ID (NG_ENB): %u\n", ue_id_e2sm.ng_enb.ng_enb_cu_ue_w1ap_id);
            break;
            
        case NG_ENB_DU_UE_ID_E2SM:
            //log_both("👤 UE ID (NG_ENB_DU): %u\n", ue_id_e2sm.ng_enb_du.ng_enb_cu_ue_w1ap_id);
            break;
            
        case EN_GNB_UE_ID_E2SM:
            //log_both("👤 UE ID (EN_GNB): %u\n", ue_id_e2sm.en_gnb.enb_ue_x2ap_id);
            break;
            
        default:
            //log_both("👤 UE ID: Unknown type: %d\n", ue_id_e2sm.type);
    }
}

// Orange 스타일 메인 콜백 함수
static void sm_cb_kpm(sm_ag_if_rd_t const* rd)
{
    assert(rd != NULL);
    assert(rd->type == INDICATION_MSG_AGENT_IF_ANS_V0);
    assert(rd->ind.type == KPM_STATS_V3_0);

    // Orange 스타일 데이터 추출
    kpm_ind_data_t const* ind = &rd->ind.kpm.ind;
    kpm_ind_msg_format_3_t const* msg_frm_3 = &ind->msg.frm_3;

    {
        lock_guard(&mtx);
        
        log_both("\n========== Indication #%d ==========\n", ++indication_counter);

        // Orange 스타일 UE별 측정값 처리
        for (size_t i = 0; i < msg_frm_3->ue_meas_report_lst_len; i++) {
            // Orange 스타일 측정값 로그
            log_kpm_measurements(&msg_frm_3->meas_report_per_ue[i].ind_msg_format_1);
        }
        
        log_both("\n");
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
        int const value = 40;
        kpm_sub.ad[0].frm_4.matching_cond_lst[0].test_info_lst = 
            filter_predicate(type, condition, value);
        
        // Action Definition Format 1
        kpm_sub.ad[0].frm_4.action_def_format_1 = fill_act_def_frm_1(report_item);
    }

    return kpm_sub;
}

int main(int argc, char *argv[]) 
{
    //log_both("🎯 === KPM Monitor xApp (Orange 기반) ===\n");
    //log_both("📡 100ms 주기로 RIC Indication 메시지 수신 및 출력\n");
    //log_both("🔥 SINR, Throughput, PRB 사용률 등 실시간 모니터링\n\n");
    
    // Signal handler 설정
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // 로그 파일 열기
    log_file = fopen("xapp_log.txt", "w");
    if (log_file == NULL) {
        log_both("⚠️  Warning: Could not create log file, output will be console only\n");
        log_to_file = false;
    } else {
        log_both("📝 Log file created: xapp_log.txt\n");
    }

    // Orange 스타일 xApp 초기화
    fr_args_t args = init_fr_args(argc, argv);
    init_xapp_api(&args);
    sleep(1);
    
    // Orange 스타일 노드 연결
    e2_node_arr_xapp_t nodes = e2_nodes_xapp_api();
    defer({ free_e2_node_arr_xapp(&nodes); });
    assert(nodes.len > 0);
    log_both("✅ Connected E2 nodes = %d\n", nodes.len);

    // Orange 스타일 mutex 초기화
    pthread_mutexattr_t attr = {0};
    int rc = pthread_mutex_init(&mtx, &attr);
    assert(rc == 0);

    // Orange 스타일 KPM subscription
    sm_ans_xapp_t* hndl = calloc(nodes.len, sizeof(sm_ans_xapp_t));
    assert(hndl != NULL);

    int const KPM_ran_function = 2;
    for (size_t i = 0; i < nodes.len; ++i) {
        e2_node_connected_xapp_t* n = &nodes.n[i];
        size_t const idx = find_sm_idx(n->rf, n->len_rf, eq_sm, KPM_ran_function);
        
        if (idx < n->len_rf && 
            n->rf[idx].defn.type == KPM_RAN_FUNC_DEF_E &&
            n->rf[idx].defn.kpm.ric_report_style_list != NULL) {
            
            //log_both("🔧 Setting up KPM subscription for node %zu...\n", i);
            kpm_sub_data_t kpm_sub = gen_kpm_subs(&n->rf[idx].defn.kpm);
            hndl[i] = report_sm_xapp_api(&n->id, KPM_ran_function, &kpm_sub, sm_cb_kpm);
            assert(hndl[i].success == true);
            free_kpm_sub_data(&kpm_sub);
            //log_both("✅ KPM subscription successful for node %zu!\n", i);
        }
    }

    //log_both("\n🎯 === KPM 실시간 모니터링 시작 ===\n");
    //log_both("📊 100ms마다 RIC Indication 메시지 수신 중...\n");
    //log_both("🛑 Press Ctrl+C to stop monitoring\n\n");

    // Orange 스타일 메인 루프 (모니터링만)
    while(monitoring_active) {
        usleep(100000); // 100ms로 더 자주 체크
        
        // Optional: 주기적 상태 출력
        static int counter = 0;
        if(++counter % 10 == 0) {
            log_both("💡 [STATUS] KPM monitoring active... (received %d indications)\n", 
                   indication_counter);
        }
    }

    // Orange 스타일 cleanup
    //log_both("\n🛑 [INFO] Stopping KPM monitor...\n");
    // cleanup 부분에 추가
    if (log_file != NULL) {
        fclose(log_file);
        //log_both("📝 Log file closed: xapp_log.txt\n");
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

    //log_both("✅ KPM Monitor xApp stopped successfully\n");
    //log_both("📊 Total indications received: %d\n", indication_counter);
    return 0;
}
