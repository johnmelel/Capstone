"""
Test Vertex AI access with current service account

This script checks if we can access Vertex AI models with the current setup.
"""
import os
from dotenv import load_dotenv

load_dotenv('../.env')

def test_vertex_access():
    """Test if Vertex AI is accessible"""
    print("="*80)
    print("TESTING VERTEX AI ACCESS")
    print("="*80)
    
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION", "us-central1")
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    print(f"\nConfiguration:")
    print(f"  Project ID: {project_id}")
    print(f"  Location: {location}")
    print(f"  Credentials: {creds_path}")
    print(f"  Exists: {os.path.exists(creds_path) if creds_path else False}")
    
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
        
        print("\n[1/3] Initializing Vertex AI...")
        vertexai.init(project=project_id, location=location)
        print("  ✓ Initialization successful")
        
        print("\n[2/3] Loading model (gemini-1.5-pro-002)...")
        model = GenerativeModel('gemini-1.5-pro-002')
        print("  ✓ Model loaded")
        
        print("\n[3/3] Testing model inference...")
        response = model.generate_content("Say 'Hello from Vertex AI!'")
        print(f"  ✓ Response: {response.text}")
        
        print("\n" + "="*80)
        print("SUCCESS! Vertex AI is accessible and working!")
        print("You can use gemini-1.5-pro-002 with 300 RPM limit!")
        print("="*80)
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n✗ Error: {error_msg}")
        
        if "403" in error_msg and "Permission" in error_msg:
            print("\n" + "="*80)
            print("IAM PERMISSION REQUIRED")
            print("="*80)
            print("\nYour service account needs the 'Vertex AI User' role.")
            print("\nAsk your project admin to run:")
            print(f"\n  gcloud projects add-iam-policy-binding {project_id} \\")
            print(f"    --member='serviceAccount:adsp-34002-ip09-team-2@{project_id}.iam.gserviceaccount.com' \\")
            print(f"    --role='roles/aiplatform.user'")
        
        print("\n" + "="*80)
        print("FALLBACK: Using AI Studio (free tier)")
        print("="*80)
        return False

if __name__ == "__main__":
    test_vertex_access()
