import re

def analyze_cluster(messages: list[str]) -> dict:
    """
    Analyzes cluster messages and extracts Event IDs and keywords to recommend a structured solution.
    """
    suggestion = {
        "category": "Unknown/Generic Error",
        "confidence": "Low",
        "likely_cause": "Unidentified anomaly sequence",
        "investigate": ["Review exact stack trace in events", "Check system logs around crash time"],
        "commands": ["Get-EventLog -LogName System -Newest 50"]
    }
    
    if not messages:
        return suggestion
        
    combined_text = " ".join(messages).lower()
    
    def has_id(id_str):
        return re.search(r'\b' + id_str + r'\b', combined_text) is not None
        
    def has_kw(*kws):
        return any(kw.lower() in combined_text for kw in kws)

    # 0. Definitive Crash Check (PRIORITY)
    if has_kw('kernel-power') or \
       has_kw('rebooted without clean shutdown') or \
       has_kw('kernel power') or \
       has_id('41'):
        return {
            "category": "Power/Unexpected Shutdown",
            "confidence": "High",
            "likely_cause": "System lost power suddenly or was hard reset. No clean shutdown occurred.",
            "investigate": [
                "Check power cable and socket stability",
                "Check battery/UPS if applicable",
                "Run chkdsk to check filesystem after dirty shutdown",
                "Review Event ID 41 details for bugcheck code"
            ],
            "commands": [
                "chkdsk C: /f",
                "Get-EventLog -LogName System -InstanceId 41 -Newest 5",
                "Get-EventLog -LogName System -InstanceId 6008 -Newest 5"
            ]
        }

    # 1. Power/Unexpected Shutdown (HIGH confidence)
    if has_id('41') or has_id('6008') or \
       has_kw('kernel power', 'unexpected shutdown', 'previous shutdown was unexpected'):
        return {
            "category": "Power/Unexpected Shutdown",
            "confidence": "High",
            "likely_cause": "The system rebooted without cleanly shutting down first. This could be caused by power loss or a hard reset.",
            "investigate": ["Inspect Power Supply Unit (PSU) stability", "Check for loose power cables", "Review Kernel-Power events for bugcheck codes"],
            "commands": ["Get-EventLog -LogName System | Where-Object {$_.EventID -in 41, 6008} -Newest 10"]
        }

    # 2. Driver Failure (HIGH confidence)
    if has_kw('fx module', 'not supported in bios') or \
       (has_kw('amd') and has_kw('driver')):
        return {
            "category": "Driver Failure",
            "confidence": "High",
            "likely_cause": "A specific hardware driver (likely AMD or BIOS-related) failed or is incompatible.",
            "investigate": ["Update AMD chipset and local GPU drivers", "Check BIOS/UEFI for 'FX Module' compatibility settings", "Rollback recent driver updates"],
            "commands": ["driverquery /v", "Get-EventLog -LogName System | Where-Object {$_.Message -like '*driver*'} -Newest 20"]
        }

    # 3. Boot/Startup Failure (MEDIUM confidence)
    if has_kw('boot configuration', 'bootmgr', 'winload'):
        return {
            "category": "Boot/Startup Failure",
            "confidence": "Medium",
            "likely_cause": "Windows failed to load essential boot files or the Boot Configuration Data (BCD).",
            "investigate": ["Run Startup Repair from Windows Recovery", "Check disk health for bad sectors", "Rebuild BCD manually"],
            "commands": ["bootrec /fixmbr", "bootrec /rebuildbcd"]
        }

    # 4. Memory Failure: ID 41 + memory keywords in messages
    if has_id('41') and has_kw('memory', 'oom', 'out of memory', 'ram'):
        return {
            "category": "Memory Failure",
            "confidence": "High",
            "likely_cause": "System rebooted unexpectedly due to memory exhaustion or failing RAM module.",
            "investigate": ["Run Windows Memory Diagnostic", "Check for memory leaks in recently installed apps"],
            "commands": ["mdsched.exe", "Get-Process | Sort-Object WS -Descending | Select-Object -First 10"]
        }
        
    # 5. Thermal/Overheating: ID 41 + thermal keywords
    if has_id('41') and has_kw('thermal', 'overheat', 'temperature', 'cooling'):
        return {
            "category": "Thermal/Overheating",
            "confidence": "High",
            "likely_cause": "System performed thermal emergency shutdown.",
            "investigate": ["Check physical cooling systems", "Verify thermal paste and fan operation in BIOS"],
            "commands": ["Get-WmiObject msacpi_thermalzonetemperature -Namespace root/wmi"]
        }

    # 6. BSOD/Kernel Stop: ID 1001 (BugCheck type)
    if has_id('1001') and has_kw('bugcheck', 'bug check'):
        return {
            "category": "BSOD/Kernel Stop",
            "confidence": "High",
            "likely_cause": "Critical kernel failure (Blue Screen of Death).",
            "investigate": ["Analyze the memory dump generated", "Check for recently updated kernel drivers"],
            "commands": ["%SystemRoot%\\Minidump", "Get-EventLog -LogName System -InstanceId 1001"]
        }

    # 7. Fault Bucket/App Crash: ID 1001 (non BugCheck)
    if has_id('1001'):
        return {
            "category": "Fault Bucket/App Crash",
            "confidence": "Medium",
            "likely_cause": "An application crashed and Windows Error Reporting generated a fault bucket.",
            "investigate": ["Check application name in the Event details", "Review recent software installations"],
            "commands": ["Get-EventLog -LogName Application -InstanceId 1001 -Newest 5"]
        }

    # 7b. Black screen / WindowsBlackScreenDiagnostics / generic fault buckets
    if has_kw('windowsblackscreendiagnostics', 'black screen') or has_kw('fault bucket', 'event name: windowsblackscreendiagnosticsv1'):
        return {
            "category": "Windows Black Screen",
            "confidence": "Medium",
            "likely_cause": "Windows recorded a black-screen diagnostics fault bucket, which usually points to a display-driver, shell startup, or graphics stack failure during session initialization.",
            "investigate": [
                "Check recent display driver, GPU utility, and Windows update changes",
                "Review Application and System logs around the black-screen timestamp for display, DWM, Explorer, or BugCheck events",
                "If the issue is repeatable, capture reliability history and any WER reports for the same fault bucket"
            ],
            "commands": [
                "Get-EventLog -LogName Application -Newest 50 | Where-Object {$_.Message -like '*BlackScreen*' -or $_.Message -like '*Fault bucket*'}",
                "Get-WinEvent -LogName System | Where-Object {$_.Message -like '*display*' -or $_.Message -like '*graphics*'} | Select-Object -First 20",
                "perfmon /rel"
            ]
        }

    # 8. Service Crash: IDs 7034, 7031
    if has_id('7034') or has_id('7031'):
        return {
            "category": "Service Crash",
            "confidence": "High",
            "likely_cause": "A Windows service terminated unexpectedly.",
            "investigate": ["Check service dependencies", "Review application logs for exactly why the service stopped"],
            "commands": ["Get-Service | Where-Object Status -eq 'Stopped'"]
        }

    # 9. Disk/NTFS Corruption: IDs 55, 98
    if has_id('55') or has_id('98'):
        return {
            "category": "Disk/NTFS Corruption",
            "confidence": "High",
            "likely_cause": "The file system structure on the disk is corrupted and unusable.",
            "investigate": ["Check SMART status of disk", "Look for physical disk errors in event logs"],
            "commands": ["chkdsk /f /r", "Get-PhysicalDisk"]
        }

    # 10. Hardware Failure: IDs 29, disk error keywords
    if has_id('29') or has_kw('disk error', 'bad block', 'io error'):
        return {
            "category": "Hardware Failure",
            "confidence": "High",
            "likely_cause": "Physical hardware fault, often related to disk or storage controller.",
            "investigate": ["Replace storage cables", "Run OEM hardware diagnostics"],
            "commands": ["wmic diskdrive get status"]
        }

    # 11. Network Failure: IDs 1014, 10010
    if has_id('1014') or has_id('10010'):
        return {
            "category": "Network Failure",
            "confidence": "High",
            "likely_cause": "DNS resolution failure or DCOM application networking issue.",
            "investigate": ["Check DNS server reachability", "Verify DCOM permissions"],
            "commands": ["ipconfig /displaydns", "ping 8.8.8.8"]
        }

    # 11. System Health / Hardware Spikes
    if has_id('9999') or has_kw('system health', 'cpu usage', 'ram usage'):
        return {
            "category": "System Resource Starvation",
            "confidence": "High",
            "likely_cause": "Critical CPU (>85%) or RAM (>90%) spike detected during analysis.",
            "investigate": ["Identify rogue background processes", "Check startup applications impacting resources"],
            "commands": ["Get-Process | Sort-Object CPU -Descending | Select-Object -First 10"]
        }

    # 12. Authentication/Logon Event: special privileges
    if has_kw('special privileges assigned'):
        return {
            "category": "Authentication/Logon Event",
            "confidence": "Low",
            "likely_cause": "Normal Windows authentication event flagged as anomaly",
            "investigate": ["This is likely a false positive", "Review if logon pattern is unusual"],
            "commands": ["Get-EventLog -LogName Security -InstanceId 4672 -Newest 20"]
        }

    return suggestion
