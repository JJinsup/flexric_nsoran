// m_reducedPmValues(true)로변경 -> report 되는 값 적어짐 (lte-enb-net-device.cc도변경했음)
// Function to Calculate the average of PRBs
double
MmWaveEnbNetDevice::CalculatePrbAverage() 
{
  double totalPrbUtilization = 0;
  auto ueMap = m_rrc->GetUeMap();

  for (auto ue : ueMap)
  {
    uint16_t rnti = ue.second->GetRnti();
    double macNumberOfSymbols = m_e2DuCalculator->GetMacNumberOfSymbolsUeSpecific(rnti, m_cellId);
    
    auto phyMac = GetMac()->GetConfigurationParameters();
    Time reportingWindow = Simulator::Now() - m_e2DuCalculator->GetLastResetTime(rnti, m_cellId);
    double denominatorPrb = std::ceil(reportingWindow.GetNanoSeconds() / 
                           phyMac->GetSlotPeriod().GetNanoSeconds()) * 14;

    if (denominatorPrb > 0)
    {
      totalPrbUtilization += (macNumberOfSymbols / denominatorPrb) * 139;
    }
  }
  NS_LOG_DEBUG("Total PRB utilization: " << totalPrbUtilization);

  long dlAvailablePrbs = 139;
  double currentPrbValue = std::min((double)(totalPrbUtilization / dlAvailablePrbs * 100), 100.0);

  // Store current value
  m_prbHistory.push_back(currentPrbValue);
  
  NS_LOG_DEBUG("Current PRB Value: " << currentPrbValue << 
               " History Size: " << m_prbHistory.size() << "/" << MAX_PRB_HISTORY);

  // Only return average when we have exactly MAX_PRB_HISTORY points
  if (m_prbHistory.size() == MAX_PRB_HISTORY)
  {
    double sum = 0;
    for (auto prb : m_prbHistory)
    {
      sum += prb;
    }
    double average = sum / MAX_PRB_HISTORY;
    
    // Remove oldest value to maintain window
    m_prbHistory.erase(m_prbHistory.begin());
    
    NS_LOG_DEBUG("Returning PRB Average: " << average);
    return average;
  }
  
  // Return -1 to indicate not enough points yet
  NS_LOG_DEBUG("Not enough points yet, returning 0");
  return 0;
}

void
MmWaveEnbNetDevice::CheckReportingFlag()
{
  NS_LOG_FUNCTION(this);
  if (!m_stopSendingMessages && m_hasValidSubscription)
  {
    const auto &sub_map = m_e2term->SubscriptionMapRef();
    if (!sub_map.empty())
    {
      try 
      {
        const auto& expr = sub_map.at("Test Condition Expression");
        const auto& value = sub_map.at("Test Condition Value");
        
        int index = std::any_cast<int>(expr);
        int threshold = std::any_cast<int>(value);

        // Get current PRB average
        double currentPrbAvg = CalculatePrbAverage();
        
        // Only check conditions if we have enough points
        if (currentPrbAvg >= 0)
        {
          bool shouldReport = MATH_CALL_BACKS[index](currentPrbAvg, threshold);
          //Jinseop 여기를 항상 TRUE처리 -> 항상 REPORT
          shouldReport = true;
          NS_LOG_DEBUG("Current PRB Average: " << currentPrbAvg << 
                       " Threshold: " << threshold << 
                       " Should Report: " << m_is_reported);
          // If we haven't started reporting yet, check if we should start
          if (!m_isReportingEnabled)
          {
            if (shouldReport)
            {
              m_is_reported = true;
              m_isReportingEnabled = true;
              BuildAndSendReportMessage(m_lastSubscriptionParams);
            }
          }
          else
          {
           // If reporting is already enabled, keep sending reports
           BuildAndSendReportMessage(m_lastSubscriptionParams);
          }
        }
      }
      catch (const std::exception& e)
      {
        NS_LOG_ERROR("Error checking PRB usage: " << e.what());
      }
    }
    // Schedule next check
    Simulator::ScheduleWithContext(1, m_checkPeriod,
        &MmWaveEnbNetDevice::CheckReportingFlag, this);
  }
}
