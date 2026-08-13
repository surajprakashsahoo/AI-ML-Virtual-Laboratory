AI/ML VIRTUAL LABORATORY — VIT VELLORE
=============================================

This package contains the Streamlit-based AI/ML Virtual Laboratory for BECE309L.

TITLE PAGE UPDATE
-----------------
- Added the supplied VIT logo above the title.
- Reworked the Home page into a concise, presentation-friendly title page.
- Added a compact module overview with seven laboratory modules.
- Added a short "Start exploring" instruction for students.
- Kept the existing interactive laboratory modules and visualizations unchanged.

FILES
-----
app.py                  Main Streamlit application
requirements.txt        Python dependencies
assets/logo.png         VIT logo used on the Home/title page

RUN LOCALLY
-----------
1. Open Command Prompt / Terminal in this folder.
2. Install dependencies:
   python -m pip install -r requirements.txt
3. Start the application:
   python -m streamlit run app.py
4. Open the displayed local URL, normally:
   http://localhost:8501

PHONE ACCESS ON THE SAME Wi-Fi
------------------------------
If the computer and phone are connected to the same Wi-Fi, start Streamlit with:

   python -m streamlit run app.py --server.address=0.0.0.0

Then find the computer's local IPv4 address using:
   Windows: ipconfig

On the phone, open:
   http://<COMPUTER-IP>:8501

For example:
   http://192.168.1.10:8501

If Windows Firewall asks for permission, allow Python/Streamlit on the
private network.

NOTE
----
The app requires the assets/logo.png file to remain in the assets folder.
