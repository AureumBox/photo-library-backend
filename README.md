# photo-library-backend

Repository intended for the backend of the photo library module for a dictionary of the Bolívar state

  
## Installing
1.  Create virtualenv with Python 3.10
    `virtualenv -p {route/to/python310} .venv`

2.  Activate .venv
     `.venv\Scripts\activate`

3. Install dependencies
    `pip install -r requirements.txt`

3. Start server
    `uvicorn app.main:app --reload`

4. Update *requirements.txt*  if necessary
    `pip freeze > requirements.txt`
