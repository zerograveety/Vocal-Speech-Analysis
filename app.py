from __future__ import annotations

import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask import jsonify
from werkzeug.utils import secure_filename

import json
try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except Exception:
    genai = None  # type: ignore
    _GEMINI_AVAILABLE = False


UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
RESULTS_FOLDER = os.path.join(UPLOAD_FOLDER, 'results')
ALLOWED_VIDEO = {'.mp4', '.mov'}
ALLOWED_PPT = {'.pptx'}


app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'dev-secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Gemini API configuration (hardcoded per user request)
GOOGLE_API_KEY = 'GEMINI_API_KEY_REMOVED'
if _GEMINI_AVAILABLE and GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)  # type: ignore[attr-defined]
    except Exception:
        _GEMINI_AVAILABLE = False

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


# -----------------------------
# Helpers to persist/load results
# -----------------------------
def _save_results_to_store(payload: dict) -> str:
    """Save results payload to JSON file and return unique result id."""
    import uuid
    result_id = str(uuid.uuid4())
    path = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return result_id


def _load_results_from_store(result_id: str) -> dict | None:
    path = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _allowed(filename: str, allowed: set[str]) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in allowed


def get_gemini_analysis(analysis_data):
    """Get Gemini analysis based on provided analysis results.

    Returns a structured dict with keys:
    - overall_score: int 0-100
    - strengths: list[str]
    - improvements: list[str]
    - recommendations: list[str]
    - summary: str (short)
    If parsing fails, returns {"summary": response_text}.
    """
    if not _GEMINI_AVAILABLE:
        # Minimal fallback: compute a tiny summary without external model
        return {
            "overall_score": 0,
            "strengths": [],
            "improvements": [],
            "recommendations": [],
            "summary": "Gemini disabled or API key not set; showing local metrics only.",
        }

    try:
        prompt = f"""
        You are evaluating a presentation.
        Input data (JSON):
        {json.dumps(analysis_data, indent=2)}

        Respond ONLY with compact JSON matching this schema (no prose outside JSON):
        {{
          "overall_score": integer  // 0-100 overall quality score
          "strengths": string[]     // 3-6 bullet points
          "improvements": string[]  // 3-6 bullet points
          "recommendations": string[] // 3-6 actionable steps
          "summary": string         // 1-2 sentence concise summary (<=220 chars)
        }}
        """
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.5-pro')  # type: ignore[attr-defined]
        
        # Generate response
        response = model.generate_content(prompt)

        text = getattr(response, 'text', '') or ''

        # Attempt to extract/parse JSON robustly
        def _extract_json(s: str) -> str:
            start = s.find('{')
            end = s.rfind('}')
            if start != -1 and end != -1 and end > start:
                return s[start:end+1]
            return s

        candidate = _extract_json(text)
        try:
            data = json.loads(candidate)
            # Coerce fields
            out = {
                "overall_score": int(max(0, min(100, int(data.get("overall_score", 0))))),
                "strengths": [str(x) for x in (data.get("strengths") or [])][:10],
                "improvements": [str(x) for x in (data.get("improvements") or [])][:10],
                "recommendations": [str(x) for x in (data.get("recommendations") or [])][:10],
                "summary": str(data.get("summary") or "")[:300],
            }
            return out
        except Exception:
            # Fallback: return truncated summary text only
            return {"summary": text[:600]}
        
    except Exception as e:
        return {"summary": f"Error getting Gemini analysis: {str(e)}"}


@app.get('/')
def index():
    return render_template('index.html')


@app.post('/analyze_video')
def analyze_video_route():
    file = request.files.get('video')
    if not file or file.filename == '':
        flash('No video file provided')
        return redirect(url_for('index'))
    if not _allowed(file.filename, ALLOWED_VIDEO):
        flash('Unsupported video type. Please upload MP4 or MOV.')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    try:
        # Lazy import to avoid heavy/brittle deps at app import time
        from video_analysis import analyze_video  # type: ignore
        video_results = analyze_video(save_path)
        gemini_analysis = get_gemini_analysis({"video_analysis": video_results})
    except Exception as e:
        flash(f'Video analysis failed: {e}')
        return redirect(url_for('index'))
    finally:
        # Keep uploads for debugging; comment next line to retain files
        pass

    # Persist results and redirect to GET route (PRG pattern)
    result_id = _save_results_to_store({
        "video_results": video_results,
        "ppt_results": None,
        "gemini_analysis": gemini_analysis,
    })
    return redirect(url_for('results_page', result_id=result_id))


@app.post('/analyze_ppt')
def analyze_ppt_route():
    file = request.files.get('ppt')
    if not file or file.filename == '':
        flash('No PPTX file provided')
        return redirect(url_for('index'))
    if not _allowed(file.filename, ALLOWED_PPT):
        flash('Unsupported file type. Please upload a PPTX file.')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    try:
        # Lazy import to avoid heavy/brittle deps at app import time
        from ppt_analysis import analyze_pptx  # type: ignore
        ppt_results = analyze_pptx(save_path)
        gemini_analysis = get_gemini_analysis({"ppt_analysis": ppt_results})
    except Exception as e:
        flash(f'PPT analysis failed: {e}')
        return redirect(url_for('index'))
    finally:
        # Keep uploads for debugging
        pass

    result_id = _save_results_to_store({
        "video_results": None,
        "ppt_results": ppt_results,
        "gemini_analysis": gemini_analysis,
    })
    return redirect(url_for('results_page', result_id=result_id))


@app.get('/analyze_existing')
def analyze_existing():
    """Analyze existing data with Gemini"""
    try:
        # Load existing analysis data
        body_file = 'outputs/Dante_dnace-body-analysis.json'
        speech_file = 'outputs/Dante_dnace-speech-analysis.json'
        
        if os.path.exists(body_file) and os.path.exists(speech_file):
            with open(body_file, 'r') as f:
                body_data = json.load(f)
            
            with open(speech_file, 'r') as f:
                speech_data = json.load(f)
            
            # Combine the data
            combined_data = {
                "body_analysis": body_data,
                "speech_analysis": speech_data
            }
            
            # Get Gemini analysis
            gemini_analysis = get_gemini_analysis(combined_data)
            
            # Persist and redirect to standard results view
            result_id = _save_results_to_store({
                "video_results": None,
                "ppt_results": None,
                "gemini_analysis": gemini_analysis,
            })
            return redirect(url_for('results_page', result_id=result_id))
        else:
            flash('No existing analysis data found')
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'Analysis failed: {e}')
        return redirect(url_for('index'))


@app.get('/results/<result_id>')
def results_page(result_id: str):
    data = _load_results_from_store(result_id)
    if not data:
        flash('Results not found or expired')
        return redirect(url_for('index'))
    return render_template('results.html', 
                           video_results=data.get('video_results'), 
                           ppt_results=data.get('ppt_results'), 
                           gemini_analysis=data.get('gemini_analysis'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4444, debug=True)

# -----------------------------
# Diagnostics
# -----------------------------
@app.get('/healthz')
def healthz():
    """Lightweight health/diagnostics endpoint for debugging video/mediapipe."""
    info: dict[str, object] = {}
    # Python/env
    import sys
    info['python'] = sys.version
    # Packages
    try:
        import cv2  # type: ignore
        info['opencv'] = getattr(cv2, '__version__', 'unknown')
    except Exception as e:  # pragma: no cover
        info['opencv_error'] = str(e)
    try:
        import mediapipe as mp  # type: ignore
        info['mediapipe'] = getattr(mp, '__version__', 'unknown')
    except Exception as e:  # pragma: no cover
        info['mediapipe_error'] = str(e)

    # Optional: attempt to open a known sample video if present
    sample = os.path.join(os.path.dirname(__file__), 'WhatsApp Video 2025-09-24 at 22.54.19.mp4')
    info['sample_exists'] = os.path.exists(sample)
    try:
        if os.path.exists(sample):
            import cv2  # type: ignore
            cap = cv2.VideoCapture(sample)
            opened = cap.isOpened()
            ok, frame = cap.read()
            cap.release()
            info['sample_opened'] = bool(opened)
            info['sample_read_first_frame'] = bool(ok)
            if opened:
                info['sample_fps'] = float(cv2.VideoCapture(sample).get(cv2.CAP_PROP_FPS))
    except Exception as e:
        info['sample_error'] = str(e)

    return jsonify(info)
