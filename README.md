# HTU Advisor — Final Reviewed Build

This package contains:

- Major recommendation model and automatic model refresh
- Rule-based eligibility checker
- Bilingual HTU chatbot
- Structured quick-access questions
- Current and archived study-plan tracking
- Course, prerequisite, tuition, admission, interview, scholarship,
  contact, deadline, and university-regulation datasets
- Automated chatbot regression tests

## Setup on Windows PowerShell

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Test before running

```powershell
python test_full_system.py
```

Expected result:

```text
PASS: 20 chatbot checks
```

## Train or refresh the recommendation model

```powershell
python train_model.py --minimum-accuracy 0.70
```

## Run the application

```powershell
python -m streamlit run app.py
```

## Data behavior

- Current study plans are used by default.
- Archived plans remain in the datasets for historical comparison.
- Duplicate documents are recorded and are not indexed twice.
- BSc, Technical, and Technician courses are kept separate.


## UI/UX improvements in this build

- Four-item primary navigation: Home, Discover, Eligibility, Advisor
- Clear value proposition and direct task cards on the home page
- Global home search that routes directly to the advisor
- Mobile-first 44–48 px touch targets
- High-contrast light and dark themes
- Visible keyboard focus states
- Reduced quick-question categories from nine to six
- Forms request only essential information
- Inline validation for missing American-certificate scores
- Loading, success, warning, and error feedback
- Current sources and confidence shown with chatbot answers
- Responsive layouts and generous whitespace


## Microphone support

The Advisor page now includes a Keyboard / Microphone input selector.

Microphone flow:

1. Choose **Microphone**.
2. Allow browser microphone permission.
3. Record and stop.
4. Review or edit the transcription.
5. Select **Send voice question**.

Voice transcription uses `SpeechRecognition` with the Google speech service and
therefore requires an internet connection. English uses `en-US`; Arabic uses
`ar-JO`.
