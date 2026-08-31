import google.generativeai as genai
import json
import os

# Configure Gemini API — read from environment variable (never hardcode keys)
GOOGLE_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GOOGLE_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GOOGLE_API_KEY)

def analyze_with_gemini():
    """Send MediaPipe analysis data to Gemini for overall analysis"""
    
    # Load the analysis data
    try:
        with open('outputs/Dante_dnace-body-analysis.json', 'r') as f:
            body_data = json.load(f)
        
        with open('outputs/Dante_dnace-speech-analysis.json', 'r') as f:
            speech_data = json.load(f)
        
        # Combine the data
        analysis_data = {
            "body_analysis": body_data,
            "speech_analysis": speech_data
        }
        
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
        6. What is the overall score of the presentation?
        7. tone, grammer, content, overall score, pacing, vocal variety, clarity, filler words, pauses, emphasis, pronunciation, overall score

        Format your response in a clear, actionable manner that would be helpful for someone looking to improve their presentation skills.
        """
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        print("Sending analysis data to Gemini...")
        print("=" * 50)
        
        # Generate response
        response = model.generate_content(prompt)
        
        print("GEMINI ANALYSIS RESULTS:")
        print("=" * 50)
        print(response.text)
        
        # Save the analysis to a file
        with open('gemini_overall_analysis.txt', 'w') as f:
            f.write("GEMINI OVERALL ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            f.write(response.text)
        
        print("\n" + "=" * 50)
        print("Analysis saved to: gemini_overall_analysis.txt")
        
        return response.text
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    analyze_with_gemini()
