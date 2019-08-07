python -m pip install virtualenv
python -m venv env
.\env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt
#Read-Host 'Press Enter to continue...' | Out-Null