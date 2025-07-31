m_allowAutonomousHoWithE2 (false) // 진섭 LteEnbRrc::LteEnbRrc ()

//LteEnbRrc::GetTypeId (void)
.AddAttribute ("AllowAutonomousHoWithE2",
              "If true, allow autonomous handover heuristics even when E2 is enabled",
              BooleanValue (false),
              MakeBooleanAccessor (&LteEnbRrc::m_allowAutonomousHoWithE2),
              MakeBooleanChecker ())

  
void
LteEnbRrc::TakeUeHoControl (uint64_t imsi)
{
  NS_LOG_FUNCTION (this << imsi);
  if (!m_allowAutonomousHoWithE2)
  {
    NS_LOG_INFO ("UE " << +imsi << " has external HO control");
    m_e2ControlledUes.insert (imsi);
  }
  else
  {
    NS_LOG_INFO ("E2 is active, but autonomous HO is allowed. Ignoring TakeUeHoControl for UE " << imsi);
  }

}

// TriggerUeAssociationUpdate 주석처리 해제
  Simulator::Schedule(MicroSeconds(m_crtPeriod), &LteEnbRrc::TriggerUeAssociationUpdate, this);  //진섭

// .h파일에 bool변수 추가
    HandoverMode m_handoverMode;
    bool m_allowAutonomousHoWithE2; // 진섭
