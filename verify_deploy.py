"""
Deployment Verification Script
Checks if the environment is ready for Cloud Run deployment
"""
import subprocess
import sys
import os

def run_check(command, name):
    print(f"Checking {name}...", end=" ")
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True, encoding='utf-8')
        if result.returncode == 0:
            print("[OK]")
            return True, result.stdout
        else:
            print("[FAIL]")
            print(f"Error: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        print(f"[EXEC ERROR]: {e}")
        return False, str(e)

def main():
    print("="*50)
    print("DEPLOYMENT PRE-FLIGHT CHECK")
    print("="*50)
    
    # 1. Check Docker
    docker_ok, _ = run_check("docker --version", "Docker")
    
    # 2. Check GCloud
    gcloud_ok, _ = run_check("gcloud --version", "Google Cloud SDK")
    
    if gcloud_ok:
        # 3. Check Auth
        print("Checking GCP Auth...", end=" ")
        auth_res = subprocess.run("gcloud auth list", capture_output=True, text=True, shell=True, encoding='utf-8')
        if "*" in auth_res.stdout:
            print("[OK] Authenticated")
        else:
            print("[WARN] Not Authenticated (Run `gcloud auth login`)")
            
        # 4. Check Project
        print("Checking GCP Project...", end=" ")
        proj_res = subprocess.run("gcloud config get-value project", capture_output=True, text=True, shell=True, encoding='utf-8')
        project = proj_res.stdout.strip()
        if project:
            print(f"[OK] Set to {project}")
        else:
            print("[FAIL] No project set (Run `gcloud config set project <PROJECT_ID>`)")

    # 5. Check dependencies present
    print("Checking critical files...", end=" ")
    critical_files = ['Dockerfile', 'requirements.txt', 'app.py', 'cloudbuild.yaml']
    missing = [f for f in critical_files if not os.path.exists(f)]
    if not missing:
        print("[OK] All present")
    else:
        print(f"[FAIL] Missing: {', '.join(missing)}")

    print("\nSUMMARY:")
    if docker_ok and gcloud_ok:
        print("[READY] Ready to deploy via 'gcloud builds submit' or 'gcloud run deploy'")
    else:
        print("[NOT READY] Environment not fully ready. See errors above.")

if __name__ == "__main__":
    main()
