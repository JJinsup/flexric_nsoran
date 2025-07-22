/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/* *
 * UE Position Prediction Optimized Scenario
 * 
 * Key Changes:
 * - 7 Base Stations (1 eNB + 6 gNB) in hexagonal layout
 * - 56 UEs for sufficient data collection
 * - ISD optimized for handover frequency
 * - Energy saving components removed
 * - Enhanced position tracking and logging
 * 
 * Copyright (c) 2024 Orange Innovation Poland
 * Copyright (c) 2024 Orange Innovation Egypt
 */
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-helper.h"
#include <ns3/lte-ue-net-device.h>
#include "ns3/mmwave-helper.h"
#include "ns3/epc-helper.h"
#include "ns3/mmwave-point-to-point-epc-helper.h"
#include "../src/mmwave/model/node-container-manager.h"
#include "ns3/lte-helper.h"
#include <sys/time.h>
#include <ctime>
#include <sys/types.h>
#include <iostream>
#include <stdlib.h>
#include <list>
#include <random>
#include <chrono>
#include <cmath>
#include <fstream>
#include "ns3/isotropic-antenna-model.h"

using namespace ns3;
using namespace mmwave;

// Position prediction related variables (removed energy saving)
std::map<uint64_t, uint16_t> imsi_cellid;
std::map<uint16_t, std::set<uint64_t>> imsi_list;
std::map<uint16_t, Ptr<Node>> cellid_node;
std::map<uint32_t, uint16_t> ue_cellid_usinghandover;
std::map<uint64_t, uint32_t> ueimsi_nodeid;
int ue_assoc_list[100] = {0}; // Increased for 56 UEs
double maxXAxis;
double maxYAxis;

NS_LOG_COMPONENT_DEFINE ("PositionPredictionScenario");

void
PrintGnuplottableUeListToFile(std::string filename) {
    std::ofstream outFile;
    outFile.open(filename.c_str(), std::ios_base::out | std::ios_base::trunc);
    if (!outFile.is_open()) {
        NS_LOG_ERROR("Can't open file " << filename);
        return;
    }
    for (NodeList::Iterator it = NodeList::Begin(); it != NodeList::End(); ++it) {
        Ptr<Node> node = *it;
        int nDevs = node->GetNDevices();
        for (int j = 0; j < nDevs; j++) {
            Ptr<LteUeNetDevice> uedev = node->GetDevice(j)->GetObject<LteUeNetDevice>();
            Ptr<MmWaveUeNetDevice> mmuedev = node->GetDevice(j)->GetObject<MmWaveUeNetDevice>();
            Ptr<McUeNetDevice> mcuedev = node->GetDevice(j)->GetObject<McUeNetDevice>();
            if (uedev) {
                Vector pos = node->GetObject<MobilityModel>()->GetPosition();
                outFile << "set label \"" << uedev->GetImsi() << "\" at " << pos.x << "," << pos.y
                        << " left font \"Helvetica,8\" textcolor rgb \"black\" front point pt 1 ps "
                           "0.3 lc rgb \"black\" offset 0,0"
                        << std::endl;
            } else if (mmuedev) {
                Vector pos = node->GetObject<MobilityModel>()->GetPosition();
                outFile << "set label \"" << mmuedev->GetImsi() << "\" at " << pos.x << "," << pos.y
                        << " left font \"Helvetica,8\" textcolor rgb \"black\" front point pt 1 ps "
                           "0.3 lc rgb \"black\" offset 0,0"
                        << std::endl;
            } else if (mcuedev) {
                Vector pos = node->GetObject<MobilityModel>()->GetPosition();
                outFile << "set label \"" << mcuedev->GetImsi() << "\" at " << pos.x << "," << pos.y
                        << " left font \"Helvetica,8\" textcolor rgb \"black\" front point pt 1 ps "
                           "0.3 lc rgb \"black\" offset 0,0"
                        << std::endl;
            }
        }
    }
}

void
PrintGnuplottableEnbListToFile(uint64_t m_startTime) {
    uint64_t timestamp = m_startTime + (uint64_t) Simulator::Now().GetMilliSeconds();
    
    std::string filename1 = "enbs.txt";
    std::string filename2 = "gnbs.txt";
    
    for (NodeList::Iterator it = NodeList::Begin(); it != NodeList::End(); ++it) {
        Ptr<Node> node = *it;
        int nDevs = node->GetNDevices();
        for (int j = 0; j < nDevs; j++) {
            Ptr<LteEnbNetDevice> enbdev = node->GetDevice(j)->GetObject<LteEnbNetDevice>();
            Ptr<MmWaveEnbNetDevice> mmdev = node->GetDevice(j)->GetObject<MmWaveEnbNetDevice>();
            if (enbdev) {
                Vector pos = node->GetObject<MobilityModel>()->GetPosition();
                std::ofstream outFile1;
                outFile1.open(filename1.c_str(), std::ios_base::out | std::ios_base::app);
                if (!outFile1.is_open()) {
                    NS_LOG_ERROR("Can't open file " << filename1);
                    return;
                }
                // Simplified output format (removed energy data)
                outFile1 << timestamp << "," << enbdev->GetCellId() << "," << pos.x << "," << pos.y << ","
                         << m_startTime << std::endl;
                outFile1.close();
            } else if (mmdev) {
                Vector pos = node->GetObject<MobilityModel>()->GetPosition();
                std::ofstream outFile2;
                outFile2.open(filename2.c_str(), std::ios_base::out | std::ios_base::app);
                if (!outFile2.is_open()) {
                    NS_LOG_ERROR("Can't open file " << filename2);
                    return;
                }
                
                // Track UE associations for position prediction
                auto ueMap = mmdev->GetUeMap();
                for (const auto &ue: ueMap) {
                    uint64_t imsi_assoc = ue.second->GetImsi();
                    ue_assoc_list[imsi_assoc - 1] = mmdev->GetCellId();
                }
                
                uint16_t cell_id = mmdev->GetCellId();
                // Simplified output format (removed energy data)
                outFile2 << timestamp << "," << cell_id << "," << pos.x << "," << pos.y << ","
                         << m_startTime << std::endl;
                outFile2.close();
            }
        }
    }
}

void
ClearFile(std::string Filename, uint64_t m_startTime) {
    std::string filename = Filename;
    std::ofstream outFile;
    outFile.open(filename.c_str(), std::ios_base::out | std::ios_base::trunc);
    if (!outFile.is_open()) {
        NS_LOG_ERROR("Can't open file " << filename);
        return;
    }
    outFile.close();
    
    uint64_t timestamp = m_startTime + (uint64_t) Simulator::Now().GetMilliSeconds();
    std::ofstream outFile1;
    outFile1.open(filename.c_str(), std::ios_base::out | std::ios_base::app);

    if (Filename == "ue_position.txt") {
        outFile1 << "timestamp,id,x,y,type,cell,simid" << std::endl;
    } else {
        outFile1 << "timestamp,id,x,y,simid" << std::endl;
        outFile1 << timestamp << "," << "0" << "," << maxXAxis << "," << maxYAxis << std::endl;
    }
    outFile1.close();
}

void
PrintPosition(Ptr<Node> node, int iterator, std::string Filename, uint64_t m_startTime) {
    uint64_t timestamp = m_startTime + (uint64_t) Simulator::Now().GetMilliSeconds();

    int imsi;
    Ptr<Node> node1 = NodeList::GetNode(iterator);
    int nDevs = node->GetNDevices();
    std::string filename = Filename;
    std::ofstream outFile;
    for (int j = 0; j < nDevs; j++) {
        Ptr<McUeNetDevice> mcuedev = node1->GetDevice(j)->GetObject<McUeNetDevice>();
        if (mcuedev) {
            imsi = int(mcuedev->GetImsi());
            int serving_cell = ue_assoc_list[imsi - 1];
            if (serving_cell == 0) {
                serving_cell = 1;
            }
            Ptr<MobilityModel> model = node->GetObject<MobilityModel>();
            Vector position = model->GetPosition();
            
            NS_LOG_UNCOND("Position of UE with IMSI " << imsi << " is " << model->GetPosition()
                                                     << " at time "
                                                     << Simulator::Now().GetSeconds()
                                                     << ", UE connected to Cell: " << serving_cell);
            
            outFile.open(filename.c_str(), std::ios_base::out | std::ios_base::app);
            if (!outFile.is_open()) {
                NS_LOG_ERROR("Can't open file " << filename);
                return;
            }

            outFile << timestamp << "," << imsi << "," << position.x << "," << position.y << ",mc,"
                    << serving_cell << "," << m_startTime << std::endl;
            outFile.close();
        }
    }
}

// Global Values for Position Prediction Scenario
static ns3::GlobalValue g_bufferSize("bufferSize", "RLC tx buffer size (MB)",
                                      ns3::UintegerValue(10),
                                      ns3::MakeUintegerChecker<uint32_t>());

static ns3::GlobalValue g_enableTraces("enableTraces", "If true, generate ns-3 traces",
                                        ns3::BooleanValue(true), ns3::MakeBooleanChecker());

static ns3::GlobalValue g_e2lteEnabled("e2lteEnabled", "If true, send LTE E2 reports",
                                        ns3::BooleanValue(true), ns3::MakeBooleanChecker());

static ns3::GlobalValue g_e2nrEnabled("e2nrEnabled", "If true, send NR E2 reports",
                                       ns3::BooleanValue(true), ns3::MakeBooleanChecker());

static ns3::GlobalValue g_e2du("e2du", "If true, send DU reports", ns3::BooleanValue(false),
                                ns3::MakeBooleanChecker());

static ns3::GlobalValue g_e2cuUp("e2cuUp", "If true, send CU-UP reports", ns3::BooleanValue(false),
                                  ns3::MakeBooleanChecker());

static ns3::GlobalValue g_e2cuCp("e2cuCp", "If true, send CU-CP reports", ns3::BooleanValue(true),
                                  ns3::MakeBooleanChecker());

static ns3::GlobalValue g_reducedPmValues("reducedPmValues", "If true, use a subset of the pm containers",
                                          ns3::BooleanValue(false), ns3::MakeBooleanChecker());

// Enhanced for position prediction
static ns3::GlobalValue g_hoSinrDifference("hoSinrDifference",
                        "The value for which an handover between MmWave eNB is triggered",
                        ns3::DoubleValue(3), ns3::MakeDoubleChecker<double>()); // Reduced for more HOs

static ns3::GlobalValue g_indicationPeriodicity("indicationPeriodicity",
                             "E2 Indication Periodicity reports (value in seconds)",
                             ns3::DoubleValue(0.1), ns3::MakeDoubleChecker<double>(0.01, 2.0));

static ns3::GlobalValue g_simTime("simTime", "Simulation time in seconds", 
                                   ns3::DoubleValue(90), // 10 minutes for sufficient data
                                   ns3::MakeDoubleChecker<double>(0.1, 100000.0));

static ns3::GlobalValue g_outageThreshold("outageThreshold",
                                           "SNR threshold for outage events [dB]",
                                           ns3::DoubleValue(-5.0),
                                           ns3::MakeDoubleChecker<double>());

static ns3::GlobalValue g_numberOfRaPreambles(
    "numberOfRaPreambles",
    "how many random access preambles are available for the contention based RACH process",
    ns3::UintegerValue(40),
    ns3::MakeUintegerChecker<uint8_t>());

static ns3::GlobalValue g_handoverMode("handoverMode",
                    "HO heuristic to be used",
                    ns3::StringValue("DynamicTtt"), ns3::MakeStringChecker());

static ns3::GlobalValue g_e2TermIp("e2TermIp", "The IP address of the RIC E2 termination",
                                    ns3::StringValue("127.0.0.1"), ns3::MakeStringChecker());

static ns3::GlobalValue g_enableE2FileLogging("enableE2FileLogging",
                              "If true, generate offline file logging instead of connecting to RIC",
                              ns3::BooleanValue(false), ns3::MakeBooleanChecker());

static ns3::GlobalValue g_e2_func_id("KPM_E2functionID", "Function ID to subscribe",
                                      ns3::DoubleValue(2),
                                      ns3::MakeDoubleChecker<double>());

static ns3::GlobalValue g_rc_e2_func_id("RC_E2functionID", "Function ID to subscribe",
                                         ns3::DoubleValue(3),
                                         ns3::MakeDoubleChecker<double>());

static ns3::GlobalValue g_controlFileName("controlFileName",
                                           "The path to the control file (can be absolute)",
                                           ns3::StringValue(""),
                                           ns3::MakeStringChecker());

// Position Prediction Optimized Parameters
static ns3::GlobalValue mmWave_nodes ("N_MmWaveEnbNodes", "Number of mmWaveNodes",
                                      ns3::UintegerValue (7), // Changed from 4 to 7
                                      ns3::MakeUintegerChecker<uint8_t> ());

static ns3::GlobalValue ue_s ("N_Ues", "Number of User Equipments",
                              ns3::UintegerValue (28), // Changed from 3 to 28
                              ns3::MakeUintegerChecker<uint32_t> ());

static ns3::GlobalValue center_freq ("CenterFrequency", "Center Frequency Value",
                                     ns3::DoubleValue (3.5e9),
                                     ns3::MakeDoubleChecker<double> ());

static ns3::GlobalValue bandwidth_value ("Bandwidth", "Bandwidth Value",
                                         ns3::DoubleValue (20e6),
                                         ns3::MakeDoubleChecker<double> ());

static ns3::GlobalValue interside_distance_value_ue ("IntersideDistanceUEs", "Interside Distance Value",
                                      ns3::DoubleValue (450), 
                                      ns3::MakeDoubleChecker<double> ());

static ns3::GlobalValue interside_distance_value_cell ("IntersideDistanceCells", "Interside Distance Value",
                                                  ns3::DoubleValue (400), // Changed from 600 to 400
                                                  ns3::MakeDoubleChecker<double> ());

int
main(int argc, char *argv[]) {
    LogComponentEnableAll(LOG_PREFIX_ALL);
    LogComponentEnable ("MmWaveEnbNetDevice", LOG_LEVEL_FUNCTION);

    // Field dimensions
    maxXAxis = 1600;
    maxYAxis = 1600;

    CommandLine cmd;
    cmd.Parse(argc, argv);

    bool harqEnabled = true;

    UintegerValue uintegerValue;
    BooleanValue booleanValue;
    StringValue stringValue;
    DoubleValue doubleValue;

    // Get configuration values
    GlobalValue::GetValueByName("hoSinrDifference", doubleValue);
    double hoSinrDifference = doubleValue.Get();
    GlobalValue::GetValueByName("bufferSize", uintegerValue);
    uint32_t bufferSize = uintegerValue.Get();
    GlobalValue::GetValueByName("enableTraces", booleanValue);
    bool enableTraces = booleanValue.Get();
    GlobalValue::GetValueByName("outageThreshold", doubleValue);
    double outageThreshold = doubleValue.Get();
    GlobalValue::GetValueByName("handoverMode", stringValue);
    std::string handoverMode = stringValue.Get();
    GlobalValue::GetValueByName("e2TermIp", stringValue);
    std::string e2TermIp = stringValue.Get();
    GlobalValue::GetValueByName("enableE2FileLogging", booleanValue);
    bool enableE2FileLogging = booleanValue.Get();
    GlobalValue::GetValueByName("KPM_E2functionID", doubleValue);
    double g_e2_func_id = doubleValue.Get();
    GlobalValue::GetValueByName("RC_E2functionID", doubleValue);
    double g_rc_e2_func_id = doubleValue.Get();
    GlobalValue::GetValueByName("numberOfRaPreambles", uintegerValue);
    uint8_t numberOfRaPreambles = uintegerValue.Get();

    NS_LOG_UNCOND("=== UE Position Prediction Scenario ===");
    NS_LOG_UNCOND("bufferSize " << bufferSize << " OutageThreshold " << outageThreshold
                                << " HandoverMode " << handoverMode << " e2TermIp " << e2TermIp
                                << " enableE2FileLogging " << enableE2FileLogging
                                << " E2 Function ID " << g_e2_func_id);

    GlobalValue::GetValueByName("e2lteEnabled", booleanValue);
    bool e2lteEnabled = booleanValue.Get();
    GlobalValue::GetValueByName("e2nrEnabled", booleanValue);
    bool e2nrEnabled = booleanValue.Get();
    GlobalValue::GetValueByName("e2du", booleanValue);
    bool e2du = booleanValue.Get();
    GlobalValue::GetValueByName("e2cuUp", booleanValue);
    bool e2cuUp = booleanValue.Get();
    GlobalValue::GetValueByName("e2cuCp", booleanValue);
    bool e2cuCp = booleanValue.Get();
    GlobalValue::GetValueByName("reducedPmValues", booleanValue);
    bool reducedPmValues = booleanValue.Get();
    GlobalValue::GetValueByName("indicationPeriodicity", doubleValue);
    double indicationPeriodicity = doubleValue.Get();
    GlobalValue::GetValueByName("controlFileName", stringValue);
    std::string controlFilename = stringValue.Get();

    NS_LOG_UNCOND("e2lteEnabled " << e2lteEnabled << " e2nrEnabled " << e2nrEnabled << " e2du "
                                 << e2du << " e2cuCp " << e2cuCp << " e2cuUp " << e2cuUp
                                 << " indicationPeriodicity " << indicationPeriodicity);

    // E2 Configuration
    Config::SetDefault("ns3::LteEnbNetDevice::ControlFileName", StringValue(controlFilename));
    Config::SetDefault("ns3::LteEnbNetDevice::E2Periodicity", DoubleValue(indicationPeriodicity));
    Config::SetDefault("ns3::MmWaveEnbNetDevice::E2Periodicity", DoubleValue(indicationPeriodicity));

    Config::SetDefault("ns3::MmWaveHelper::E2ModeLte", BooleanValue(e2lteEnabled));
    Config::SetDefault("ns3::MmWaveHelper::E2ModeNr", BooleanValue(e2nrEnabled));

    Config::SetDefault("ns3::MmWaveEnbNetDevice::EnableDuReport", BooleanValue(e2du));
    Config::SetDefault("ns3::MmWaveEnbNetDevice::EnableCuUpReport", BooleanValue(e2cuUp));
    Config::SetDefault("ns3::LteEnbNetDevice::EnableCuUpReport", BooleanValue(e2cuUp));
    Config::SetDefault("ns3::MmWaveEnbNetDevice::EnableCuCpReport", BooleanValue(e2cuCp));
    Config::SetDefault("ns3::LteEnbNetDevice::EnableCuCpReport", BooleanValue(e2cuCp));

    Config::SetDefault("ns3::MmWaveEnbNetDevice::ReducedPmValues", BooleanValue(reducedPmValues));
    Config::SetDefault("ns3::LteEnbNetDevice::ReducedPmValues", BooleanValue(reducedPmValues));

    Config::SetDefault("ns3::LteEnbNetDevice::EnableE2FileLogging", BooleanValue(enableE2FileLogging));
    Config::SetDefault("ns3::MmWaveEnbNetDevice::EnableE2FileLogging", BooleanValue(enableE2FileLogging));

    Config::SetDefault("ns3::LteEnbNetDevice::KPM_E2functionID", DoubleValue(g_e2_func_id));
    Config::SetDefault("ns3::MmWaveEnbNetDevice::KPM_E2functionID", DoubleValue(g_e2_func_id));
    Config::SetDefault("ns3::LteEnbNetDevice::RC_E2functionID", DoubleValue(g_rc_e2_func_id));

    Config::SetDefault("ns3::MmWaveEnbMac::NumberOfRaPreambles", UintegerValue(numberOfRaPreambles));

    Config::SetDefault("ns3::MmWaveHelper::HarqEnabled", BooleanValue(harqEnabled));
    Config::SetDefault("ns3::MmWaveHelper::UseIdealRrc", BooleanValue(true));
    Config::SetDefault("ns3::MmWaveHelper::E2TermIp", StringValue(e2TermIp));

    Config::SetDefault("ns3::MmWaveFlexTtiMacScheduler::HarqEnabled", BooleanValue(harqEnabled));
    Config::SetDefault("ns3::MmWavePhyMacCommon::NumHarqProcess", UintegerValue(100));

    Config::SetDefault("ns3::PhasedArrayModel::AntennaElement",
        PointerValue(CreateObject<IsotropicAntennaModel>()));
    Config::SetDefault ("ns3::ThreeGppChannelModel::UpdatePeriod", TimeValue (MilliSeconds (100.0)));
    Config::SetDefault ("ns3::ThreeGppChannelConditionModel::UpdatePeriod", TimeValue (MilliSeconds (100)));

    Config::SetDefault("ns3::LteRlcAm::ReportBufferStatusTimer", TimeValue(MilliSeconds(10.0)));
    Config::SetDefault("ns3::LteRlcUmLowLat::ReportBufferStatusTimer", TimeValue(MilliSeconds(10.0)));
    Config::SetDefault("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue(bufferSize * 1024 * 1024));
    Config::SetDefault("ns3::LteRlcUmLowLat::MaxTxBufferSize", UintegerValue(bufferSize * 1024 * 1024));
    Config::SetDefault("ns3::LteRlcAm::MaxTxBufferSize", UintegerValue(bufferSize * 1024 * 1024));

    // Enhanced handover settings for position prediction
    Config::SetDefault("ns3::LteEnbRrc::OutageThreshold", DoubleValue(outageThreshold));
    Config::SetDefault("ns3::LteEnbRrc::SecondaryCellHandoverMode", StringValue(handoverMode));
    Config::SetDefault("ns3::LteEnbRrc::HoSinrDifference", DoubleValue(hoSinrDifference));
    Config::SetDefault("ns3::ThreeGppPropagationLossModel::Frequency", DoubleValue(3.5e9));
    Config::SetDefault("ns3::ThreeGppPropagationLossModel::ShadowingEnabled", BooleanValue(false));

    // Network parameters
    double bandwidth = 20e6;
    double centerFrequency = 3.5e9;
    
    // Position prediction optimized parameters
    GlobalValue::GetValueByName("IntersideDistanceUEs", doubleValue);
    double isd_ue = doubleValue.Get(); // 320m
    GlobalValue::GetValueByName("IntersideDistanceCells", doubleValue);
    double isd_cell = doubleValue.Get(); // 400m
    
    int numAntennasMcUe = 1;
    int numAntennasMmWave = 1;

    Config::SetDefault("ns3::McUeNetDevice::AntennaNum", UintegerValue(numAntennasMcUe));
    Config::SetDefault("ns3::MmWaveNetDevice::AntennaNum", UintegerValue(numAntennasMmWave));
    Config::SetDefault("ns3::MmWavePhyMacCommon::Bandwidth", DoubleValue(bandwidth));
    Config::SetDefault("ns3::MmWavePhyMacCommon::CenterFreq", DoubleValue(centerFrequency));

    Ptr<MmWaveHelper> mmwaveHelper = CreateObject<MmWaveHelper>();
    mmwaveHelper->SetPathlossModelType("ns3::ThreeGppUmiStreetCanyonPropagationLossModel");
    mmwaveHelper->SetChannelConditionModelType("ns3::ThreeGppUmiStreetCanyonChannelConditionModel");

    Ptr<MmWavePointToPointEpcHelper> epcHelper = CreateObject<MmWavePointToPointEpcHelper>();
    mmwaveHelper->SetEpcHelper(epcHelper);

    // Enhanced network topology: 7 base stations total
    GlobalValue::GetValueByName("N_MmWaveEnbNodes", uintegerValue);
    uint8_t nMmWaveEnbNodes = uintegerValue.Get(); // 6 gNBs
    uint8_t nLteEnbNodes = 1; // 1 eNB
    GlobalValue::GetValueByName("N_Ues", uintegerValue);
    uint32_t nUeNodes = uintegerValue.Get(); // 56 UEs

    NS_LOG_UNCOND("=== Network Configuration ===");
    NS_LOG_UNCOND("mmWave gNBs: " << unsigned(nMmWaveEnbNodes));
    NS_LOG_UNCOND("LTE eNBs: " << unsigned(nLteEnbNodes));
    NS_LOG_UNCOND("Total Base Stations: " << unsigned(nMmWaveEnbNodes + nLteEnbNodes));
    NS_LOG_UNCOND("UEs: " << nUeNodes);
    NS_LOG_UNCOND("Cell ISD: " << isd_cell << "m");
    NS_LOG_UNCOND("UE Distribution Radius: " << isd_ue << "m");

    // Network setup
    Ptr<Node> pgw = epcHelper->GetPgwNode();
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);
    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(2500));
    p2ph.SetChannelAttribute("Delay", TimeValue(Seconds(0.010)));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);
    Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress(1);
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    NodeContainer ueNodes;
    NodeContainer mmWaveEnbNodes;
    NodeContainer lteEnbNodes;
    NodeContainer allEnbNodes;
    mmWaveEnbNodes.Create(nMmWaveEnbNodes);
    lteEnbNodes.Create(nLteEnbNodes);
    ueNodes.Create(nUeNodes);
    allEnbNodes.Add(lteEnbNodes);
    allEnbNodes.Add(mmWaveEnbNodes);

    NodeContainerManager::GetInstance().SetMmWaveEnbNodes(mmWaveEnbNodes);

    // Enhanced hexagonal base station positioning
    Vector centerPosition = Vector(maxXAxis / 2, maxYAxis / 2, 3);

    Ptr<ListPositionAllocator> enbPositionAlloc = CreateObject<ListPositionAllocator>();

    // Central position: LTE eNB and first mmWave gNB co-located
    enbPositionAlloc->Add(centerPosition);
    enbPositionAlloc->Add(centerPosition);
    
    NS_LOG_UNCOND("=== Base Station Positions ===");
    NS_LOG_UNCOND("eNB 1 & gNB 2: (" << centerPosition.x << ", " << centerPosition.y << ")");

    // Place 6 mmWave gNBs in hexagonal pattern around center
    for (uint8_t i = 0; i < (nMmWaveEnbNodes - 1); ++i) { // 6번반복
        double angle = (i * 60.0) * M_PI / 180.0; // FIXED: 60 degrees apart for 6 stations
        double x = centerPosition.x + isd_cell * cos(angle);
        double y = centerPosition.y + isd_cell * sin(angle);
        enbPositionAlloc->Add(Vector(x, y, 3));
        NS_LOG_UNCOND("gNB " << unsigned(i+3) << ": (" << x << ", " << y << ")");
    }

    MobilityHelper enbmobility;
    enbmobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    enbmobility.SetPositionAllocator(enbPositionAlloc);
    enbmobility.Install(allEnbNodes);

    // Enhanced UE mobility for position prediction
    MobilityHelper uemobility;
    Ptr<UniformDiscPositionAllocator> uePositionAlloc = CreateObject<UniformDiscPositionAllocator>();
    uePositionAlloc->SetX(centerPosition.x);
    uePositionAlloc->SetY(centerPosition.y);
    uePositionAlloc->SetRho(isd_ue); // 380m radius

    // Diverse speed distribution for better position prediction data
    Ptr<UniformRandomVariable> speed = CreateObject<UniformRandomVariable>();
    speed->SetAttribute("Min", DoubleValue(2.0));  // 2 m/s (pedestrian)
    speed->SetAttribute("Max", DoubleValue(5.0)); // 5 m/s (Runner)

    uemobility.SetMobilityModel("ns3::RandomWalk2dOutdoorMobilityModel", "Speed",
                                PointerValue(speed), "Bounds",
                                RectangleValue(Rectangle(0, maxXAxis, 0, maxYAxis)));
    uemobility.SetPositionAllocator(uePositionAlloc);
    uemobility.Install(ueNodes);

    NS_LOG_UNCOND("=== Mobility Configuration ===");
    NS_LOG_UNCOND("UE Speed Range: 2-5 m/s");

    // Install devices
    NetDeviceContainer lteEnbDevs = mmwaveHelper->InstallLteEnbDevice(lteEnbNodes);
    NetDeviceContainer mmWaveEnbDevs = mmwaveHelper->InstallEnbDevice(mmWaveEnbNodes);
    NetDeviceContainer mcUeDevs = mmwaveHelper->InstallMcUeDevice(ueNodes);

    // Network configuration
    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIpIface;
    ueIpIface = epcHelper->AssignUeIpv4Address(NetDeviceContainer(mcUeDevs));
    
    for (uint32_t u = 0; u < ueNodes.GetN(); ++u) {
        Ptr<Node> ueNode = ueNodes.Get(u);
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    // Add X2 interfaces for handover support
    mmwaveHelper->AddX2Interface(lteEnbNodes, mmWaveEnbNodes);
    /*
    // Add X2 interfaces between mmWave gNBs for enhanced handover
    for (uint16_t i = 0; i < mmWaveEnbNodes.GetN(); ++i) {
        for (uint16_t j = i+1; j < mmWaveEnbNodes.GetN(); ++j) {
            mmwaveHelper->AddX2Interface(mmWaveEnbNodes.Get(i), mmWaveEnbNodes.Get(j));
        }
    }
    */
    // Attach UEs to closest base station
    mmwaveHelper->AttachToClosestEnb(mcUeDevs, mmWaveEnbDevs, lteEnbDevs);

    // Application setup for data generation
    uint16_t portUdp = 60000;
    Address sinkLocalAddressUdp(InetSocketAddress(Ipv4Address::GetAny(), portUdp));
    PacketSinkHelper sinkHelperUdp("ns3::UdpSocketFactory", sinkLocalAddressUdp);
    AddressValue serverAddressUdp(InetSocketAddress(remoteHostAddr, portUdp));

    ApplicationContainer sinkApp;
    sinkApp.Add(sinkHelperUdp.Install(remoteHost));

    ApplicationContainer clientApp;

    for (uint32_t u = 0; u < ueNodes.GetN(); ++u) {
        PacketSinkHelper dlPacketSinkHelper("ns3::UdpSocketFactory",
                                           InetSocketAddress(Ipv4Address::GetAny(), 1234));
        sinkApp.Add(dlPacketSinkHelper.Install(ueNodes.Get(u)));
        UdpClientHelper dlClient(ueIpIface.GetAddress(u), 1234);
        dlClient.SetAttribute("Interval", TimeValue(MicroSeconds(500)));
        dlClient.SetAttribute("MaxPackets", UintegerValue(UINT32_MAX));
        dlClient.SetAttribute("PacketSize", UintegerValue(200)); // Small packets for frequent updates
        clientApp.Add(dlClient.Install(remoteHost));
    }

    // Start applications
    GlobalValue::GetValueByName("simTime", doubleValue);
    double simTime = doubleValue.Get();
    sinkApp.Start(Seconds(0));
    clientApp.Start(MilliSeconds(100));
    clientApp.Stop(Seconds(simTime - 0.1));

    // Enhanced position tracking and logging
    struct timeval time_now{};
    gettimeofday(&time_now, nullptr);
    uint64_t t_startTime_simid = (time_now.tv_sec * 1000) + (time_now.tv_usec / 1000);
    
    std::string ue_pos_out = "ue_position.txt";
    ClearFile(ue_pos_out, t_startTime_simid);
    ClearFile("enbs.txt", t_startTime_simid);
    ClearFile("gnbs.txt", t_startTime_simid);
    
    PrintGnuplottableUeListToFile("ues.txt");

    int nodecount = int(NodeList::GetNNodes());
    int UE_iterator = nodecount - int(nUeNodes);

    // Frequent position logging for position prediction (every 3 seconds)
    int numPrints = int(simTime / 0.1); // Every 0.1 seconds
    NS_LOG_UNCOND("=== Logging Configuration ===");
    NS_LOG_UNCOND("Position samples: " << numPrints << " (every 0.1 seconds)");
    NS_LOG_UNCOND("Total measurement points: " << numPrints * nUeNodes);

    for (int i = 1; i <= numPrints; i++) {
        double scheduleTime = i * 0.1; // 0.1초마다
        Simulator::Schedule(Seconds(scheduleTime), &PrintGnuplottableEnbListToFile, t_startTime_simid);
        for (uint32_t j = 0; j < ueNodes.GetN(); j++) {
            Simulator::Schedule(Seconds(scheduleTime + 0.01), &PrintPosition, ueNodes.Get(j), // 0.01초 후
                            j + UE_iterator, ue_pos_out, t_startTime_simid);
        }
    }

    if (enableTraces) {
        mmwaveHelper->EnableTraces();
    }

    // Enable PHY and MAC traces for detailed analysis
    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    lteHelper->Initialize();
    lteHelper->EnablePhyTraces();
    lteHelper->EnableMacTraces();

    // Run simulation
    NS_LOG_UNCOND("=== Starting Position Prediction Simulation ===");
    NS_LOG_UNCOND("Simulation time: " << simTime << " seconds");
    NS_LOG_UNCOND("Expected data points for ML training: " << numPrints * nUeNodes);
    
    Simulator::Stop(Seconds(simTime));
    NS_LOG_INFO("Run Simulation.");
    Simulator::Run();

    NS_LOG_UNCOND("=== Simulation Completed ===");
    NS_LOG_UNCOND("Position data saved to: " << ue_pos_out);
    NS_LOG_UNCOND("Base station data saved to: gnbs.txt, enbs.txt");
    NS_LOG_UNCOND("UE layout saved to: ues.txt");

    Simulator::Destroy();
    NS_LOG_INFO("Done.");
    return 0;
}
