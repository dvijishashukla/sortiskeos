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

    # 1. Memory Failure: ID 41 + memory keywords in messages
    if has_id('41') and has_kw('memory', 'oom', 'out of memory', 'ram'):
        return {
            "category": "Memory Failure",
            "confidence": "High",
            "likely_cause": "System rebooted unexpectedly due to memory exhaustion or failing RAM module.",
            "investigate": ["Run Windows Memory Diagnostic", "Check for memory leaks in recently installed apps"],
            "commands": ["mdsched.exe", "Get-Process | Sort-Object WS -Descending | Select-Object -First 10"]
        }
        
    # 2. Thermal/Overheating: ID 41 + thermal keywords
    if has_id('41') and has_kw('thermal', 'overheat', 'temperature', 'cooling'):
        return {
            "category": "Thermal/Overheating",
            "confidence": "High",
            "likely_cause": "System performed thermal emergency shutdown.",
            "investigate": ["Check physical cooling systems", "Verify thermal paste and fan operation in BIOS"],
            "commands": ["Get-WmiObject msacpi_thermalzonetemperature -Namespace root/wmi"]
        }

    # 3. Power/Unexpected Shutdown: IDs 41, 6008
    if has_id('41') or has_id('6008'):
        return {
            "category": "Power/Unexpected Shutdown",
            "confidence": "High",
            "likely_cause": "Sudden power loss or improper shutdown.",
            "investigate": ["Inspect PSU stability", "Check UPS logs if applicable"],
            "commands": ["Get-EventLog -LogName System | Where-Object {$_.EventID -in 41, 6008} -Newest 10"]
        }

    # 4. BSOD/Kernel Stop: ID 1001 (BugCheck type)
    if has_id('1001') and has_kw('bugcheck', 'bug check'):
        return {
            "category": "BSOD/Kernel Stop",
            "confidence": "High",
            "likely_cause": "Critical kernel failure (Blue Screen of Death).",
            "investigate": ["Analyze the memory dump generated", "Check for recently updated kernel drivers"],
            "commands": ["%SystemRoot%\\Minidump", "Get-EventLog -LogName System -InstanceId 1001"]
        }

    # 5. Fault Bucket/App Crash: ID 1001 (non BugCheck)
    if has_id('1001'):
        return {
            "category": "Fault Bucket/App Crash",
            "confidence": "Medium",
            "likely_cause": "An application crashed and Windows Error Reporting generated a fault bucket.",
            "investigate": ["Check application name in the Event details", "Review recent software installations"],
            "commands": ["Get-EventLog -LogName Application -InstanceId 1001 -Newest 5"]
        }

    # 6. Driver Failure: IDs 7034, 7031 with .sys in message
    if (has_id('7034') or has_id('7031')) and has_kw('.sys'):
        return {
            "category": "Driver Failure",
            "confidence": "High",
            "likely_cause": "A kernel mode driver (.sys) service terminated unexpectedly.",
            "investigate": ["Identify the offending .sys file", "Rollback or update device driver"],
            "commands": ["driverquery /v", "Get-EventLog -LogName System | Where-Object {$_.EventID -in 7034,7031}"]
        }

    # 7. Service Crash: IDs 7034, 7031 with service name
    if has_id('7034') or has_id('7031'):
        return {
            "category": "Service Crash",
            "confidence": "High",
            "likely_cause": "A Windows service terminated unexpectedly.",
            "investigate": ["Check service dependencies", "Review application logs for exactly why the service stopped"],
            "commands": ["Get-Service | Where-Object Status -eq 'Stopped'"]
        }

    # 8. Disk/NTFS Corruption: IDs 55, 98
    if has_id('55') or has_id('98'):
        return {
            "category": "Disk/NTFS Corruption",
            "confidence": "High",
            "likely_cause": "The file system structure on the disk is corrupted and unusable.",
            "investigate": ["Check SMART status of disk", "Look for physical disk errors in event logs"],
            "commands": ["chkdsk /f /r", "Get-PhysicalDisk"]
        }

    # 9. Hardware Failure: IDs 29, disk error keywords
    if has_id('29') or has_kw('disk error', 'bad block', 'io error'):
        return {
            "category": "Hardware Failure",
            "confidence": "High",
            "likely_cause": "Physical hardware fault, often related to disk or storage controller.",
            "investigate": ["Replace storage cables", "Run OEM hardware diagnostics"],
            "commands": ["wmic diskdrive get status"]
        }

    # 10. Network Failure: IDs 1014, 10010
    if has_id('1014') or has_id('10010'):
        return {
            "category": "Network Failure",
            "confidence": "High",
            "likely_cause": "DNS resolution failure or DCOM application networking issue.",
            "investigate": ["Check DNS server reachability", "Verify DCOM permissions"],
            "commands": ["ipconfig /displaydns", "ping 8.8.8.8"]
        }

    return suggestion
