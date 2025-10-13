import google.generativeai as genai
import logging
import sys
import traceback

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Your API key
GOOGLE_API_KEY = 'GEMINI_API_KEY_REMOVED'

def test_api_connection():
    try:
        print("\n=== Starting API Test ===")
        print(f"Using API Key: {GOOGLE_API_KEY[:8]}...{GOOGLE_API_KEY[-4:]}")
        
        # Configure the API
        print("\n1. Configuring API...")
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("API configured successfully")
        
        # Get available models
        print("\n2. Fetching available models...")
        models = genai.list_models()
        if not models:
            logger.error("No models available")
            print("❌ API Test Failed: No models available")
            return False
            
        # Print available models
        print("\nAvailable models:")
        for model in models:
            print(f"- {model.name}")
        
        # Try to use the first available Gemini model
        print("\n3. Looking for Gemini models...")
        model_name = None
        for model in models:
            if "gemini" in model.name.lower():
                model_name = model.name
                print(f"Found Gemini model: {model_name}")
                break
                
        if not model_name:
            logger.error("No Gemini models found")
            print("❌ API Test Failed: No Gemini models found")
            return False
            
        # Test the connection
        print(f"\n4. Testing connection with model: {model_name}")
        model = genai.GenerativeModel(model_name)
        print("Sending test request...")
        response = model.generate_content("Hello, this is a test.")
        
        if response:
            logger.info("Successfully received response from API")
            print("\nAPI Test Results:")
            print(f"✅ API configuration successful (using model: {model_name})")
            print("✅ Model connection successful")
            print("✅ Response received successfully")
            print(f"\nTest response: {response.text}")
            return True
        else:
            logger.error("No response received from API")
            print("❌ API Test Failed: No response received")
            return False
            
    except Exception as e:
        logger.error(f"API test failed: {str(e)}")
        print("\n❌ API Test Failed:")
        print(f"Error: {str(e)}")
        print("\nDetailed error information:")
        print(traceback.format_exc())
        print("\nTroubleshooting steps:")
        print("1. Verify your API key is correct")
        print("2. Check if you have enabled the Gemini API in Google Cloud Console")
        print("3. Ensure you have sufficient quota available")
        print("4. Verify your API key has the necessary permissions")
        print("5. Check your internet connection")
        print("6. Try using a different network if possible")
        return False

def analyze_mediapipe_data_with_gemini():
    """Send MediaPipe analysis data to Gemini for overall analysis"""
    try:
        import json
        
        # Load the analysis data
        with open('outputs/Dante_dnace-body-analysis.json', 'r') as f:
            body_data = json.load(f)
        
        with open('outputs/Dante_dnace-speech-analysis.json', 'r') as f:
            speech_data = json.load(f)
        
        # Configure the API
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Prepare the prompt for Gemini
        prompt = f"""
        Based on the following MediaPipe video analysis data, provide a comprehensive overall analysis and feedback:

        BODY LANGUAGE ANALYSIS:
        {json.dumps(body_data, indent=2)}

        SPEECH ANALYSIS:
        {json.dumps(speech_data, indent=2)}

        Please provide:
        1. Overall presentation quality assessment (1-10 scale)
        2. Key strengths identified
        3. Areas for improvement
        4. Specific recommendations for enhancement
        5. Professional feedback on presentation skills

        Format your response in a clear, actionable manner that would be helpful for someone looking to improve their presentation skills.
        """
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        print("\n" + "="*60)
        print("SENDING MEDIAPIPE ANALYSIS DATA TO GEMINI")
        print("="*60)
        
        # Generate response
        response = model.generate_content(prompt)
        
        print("\nGEMINI OVERALL ANALYSIS:")
        print("="*60)
        print(response.text)
        
        # Save the analysis to a file
        with open('gemini_overall_analysis.txt', 'w') as f:
            f.write("GEMINI OVERALL ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            f.write(response.text)
        
        print("\n" + "="*60)
        print("Analysis saved to: gemini_overall_analysis.txt")
        print("="*60)
        
        return response.text
        
    except Exception as e:
        print(f"Error analyzing with Gemini: {str(e)}")
        return None

if __name__ == "__main__":
    # First test the API connection
    if test_api_connection():
        print("\n" + "="*60)
        print("API CONNECTION SUCCESSFUL - PROCEEDING WITH ANALYSIS")
        print("="*60)
        # Then analyze the MediaPipe data
        analyze_mediapipe_data_with_gemini()
    else:
        print("\nAPI connection failed. Cannot proceed with analysis.") 