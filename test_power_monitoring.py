#!/usr/bin/env python3
"""
Test script to check available power monitoring methods on the system.
"""

import os
import subprocess
import glob
import time

def test_ipmi():
    """Test IPMI power monitoring availability."""
    print("Testing IPMI power monitoring...")
    try:
        # Test ipmitool availability
        result = subprocess.run(['ipmitool', 'sdr', 'type', 'Current'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ IPMI available - can read system power")
            print("Sample output:")
            for line in result.stdout.split('\n')[:3]:  # Show first 3 lines
                if line.strip():
                    print(f"  {line}")
            return True
        else:
            # Try alternative command
            result = subprocess.run(['ipmitool', 'sensor', 'get', 'System Power'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✓ IPMI available - can read system power (alternative method)")
                return True
            else:
                print("✗ IPMI not available or no power sensors found")
                return False
                
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("✗ IPMI tools not installed or not accessible")
        return False

def test_rapl():
    """Test RAPL power monitoring availability."""
    print("\nTesting RAPL power monitoring...")
    
    rapl_paths = [
        "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
        "/sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj",
        "/sys/class/powercap/intel-rapl:0/energy_uj",
        "/sys/class/powercap/intel-rapl:1/energy_uj",
    ]
    
    dram_paths = [
        "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj",
        "/sys/class/powercap/intel-rapl/intel-rapl:1/intel-rapl:1:0/energy_uj",
    ]
    
    rapl_available = False
    dram_available = False
    
    for path in rapl_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    energy = int(f.read().strip())
                    print(f"✓ RAPL CPU/Package power available: {path}")
                    rapl_available = True
                    break
            except (IOError, ValueError):
                continue
    
    if not rapl_available:
        print("✗ RAPL CPU power monitoring not available")
    
    for path in dram_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    energy = int(f.read().strip())
                    print(f"✓ RAPL DRAM power available: {path}")
                    dram_available = True
                    break
            except (IOError, ValueError):
                continue
    
    if not dram_available:
        print("✗ RAPL DRAM power monitoring not available")
    
    return rapl_available or dram_available

def test_hwmon():
    """Test hwmon power monitoring availability."""
    print("\nTesting hwmon power monitoring...")
    
    hwmon_paths = []
    for hwmon_dir in glob.glob('/sys/class/hwmon/hwmon*/'):
        power_files = glob.glob(f"{hwmon_dir}power*_input")
        hwmon_paths.extend(power_files)
    
    if hwmon_paths:
        print(f"✓ Found {len(hwmon_paths)} hwmon power sensors:")
        for path in hwmon_paths[:5]:  # Show first 5
            try:
                with open(path, 'r') as f:
                    power_uw = int(f.read().strip())
                    power_w = power_uw / 1000000
                    print(f"  {path}: {power_w:.2f}W")
            except (IOError, ValueError):
                print(f"  {path}: [Error reading]")
        
        if len(hwmon_paths) > 5:
            print(f"  ... and {len(hwmon_paths) - 5} more")
        return True
    else:
        print("✗ No hwmon power sensors found")
        return False

def test_gpu_power():
    """Test GPU power monitoring availability."""
    print("\nTesting GPU power monitoring...")
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            power_str = result.stdout.strip()
            if power_str and power_str != 'N/A':
                print(f"✓ GPU power monitoring available: {power_str}W")
                return True
            else:
                print("✗ GPU present but power monitoring not available")
                return False
        else:
            print("✗ nvidia-smi failed")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("✗ nvidia-smi not available")
        return False

def test_permissions():
    """Test file permissions for power monitoring."""
    print("\nTesting file permissions...")
    
    test_files = [
        "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
        "/sys/class/powercap/intel-rapl:0/energy_uj",
    ]
    
    permission_ok = True
    for path in test_files:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    f.read()
                print(f"✓ Can read {path}")
            except IOError as e:
                print(f"✗ Cannot read {path}: {e}")
                permission_ok = False
            break
    
    if not permission_ok:
        print("\nTo fix permission issues, try:")
        print("  sudo chmod -R 644 /sys/class/powercap/intel-rapl/*/energy_uj")
        print("  Or run the benchmark with sudo")
    
    return permission_ok

def main():
    print("Power Monitoring Capability Test")
    print("=" * 40)
    
    methods_available = []
    
    if test_ipmi():
        methods_available.append("IPMI (total system power)")
    
    if test_rapl():
        methods_available.append("RAPL (CPU/package/DRAM power)")
    
    if test_hwmon():
        methods_available.append("hwmon sensors")
    
    if test_gpu_power():
        methods_available.append("GPU power (nvidia-smi)")
    
    test_permissions()
    
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    
    if methods_available:
        print("Available power monitoring methods:")
        for method in methods_available:
            print(f"  ✓ {method}")
        
        print(f"\nRecommendation:")
        if "IPMI (total system power)" in methods_available:
            print("  Use IPMI for most accurate total system power measurement")
        elif "hwmon sensors" in methods_available:
            print("  Use hwmon sensors for system power measurement")
        elif "RAPL (CPU/package/DRAM power)" in methods_available:
            print("  Use RAPL for CPU/package power (total system power not available)")
        
        print("\nThe benchmark will automatically use the best available method.")
        
    else:
        print("✗ No power monitoring methods available")
        print("\nTo enable power monitoring:")
        print("  1. Install IPMI tools: sudo apt install ipmitool")
        print("  2. Install sensors: sudo apt install lm-sensors")
        print("  3. Check hardware support for RAPL")
        print("  4. Run with elevated permissions if needed")

if __name__ == "__main__":
    main() 