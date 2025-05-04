# **Overview**

## **What It Is**

The Video Feature Tracking App is a web application that allows users to track points of interest in video files. Built with a FastAPI backend and Streamlit frontend, it provides an intuitive interface for video analysis.

## **What It Does**

- This application enables users to:

- Upload MP4 video files

- Select specific points within the video frame to track

- Choose a time range for analysis

- Apply various preprocessing options:

  - Image stabilization

  - Histogram equalization

  - Gaussian blur

  - Edge detection

- View tracking results with:

  - Frame-by-frame navigation

  - Speed analysis graphs

  - Position tracking visualization


The app uses the Lucas-Kanade optical flow method from OpenCV to accurately track point movement between frames, even when objects change position, lighting, or appearance.

## **How to Access It**

**The application is deployed on Google Cloud Run and can be accessed at:**

Frontend: https://frontend-z5ayxyaklq-uc.a.run.app

Simply navigate to the frontend URL in your browser to start using the application. No account creation or login is required.

## **Local Setup (Optional)**

**Install the required dependencies:**

pip install -r requirements.txt

**Run the Streamlit frontend:**

streamlit run app.py

**By default, the app will connect to the cloud-hosted backend, but you can also run the backend locally:**

uvicorn main:app --host 0.0.0.0 --port 8000
