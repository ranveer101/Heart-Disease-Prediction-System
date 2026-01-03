# Deployment Guide for Render

## Files Created for Deployment

1. **requirements.txt** - Lists all Python dependencies
2. **Procfile** - Tells Render how to start your application
3. **render.yaml** - Optional configuration file for Render
4. **.gitignore** - Excludes unnecessary files from Git
5. **app.py** - Updated to use environment variables

## Prerequisites

1. **MongoDB Database**: You need a MongoDB database. Options:
   - MongoDB Atlas (free tier available) - Recommended
   - Render MongoDB service
   - Any cloud MongoDB provider

2. **GitHub Account**: Render deploys from GitHub repositories

## Step-by-Step Deployment Instructions

### Step 1: Set Up MongoDB

1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) and create a free account
2. Create a new cluster (free tier M0)
3. Create a database user (username and password)
4. Add your IP address to the whitelist (or use 0.0.0.0/0 for all IPs - less secure but easier)
5. Get your connection string:
   - Click "Connect" on your cluster
   - Choose "Connect your application"
   - Copy the connection string (it looks like: `mongodb+srv://username:password@cluster.mongodb.net/`)
   - Replace `<password>` with your actual password
   - Add database name at the end: `mongodb+srv://username:password@cluster.mongodb.net/heart_disease_db`

### Step 2: Push Code to GitHub

1. Initialize Git repository (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. Create a new repository on GitHub

3. Push your code:
   ```bash
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git branch -M main
   git push -u origin main
   ```

### Step 3: Deploy on Render

1. Go to [Render](https://render.com) and sign up/login

2. Click "New +" → "Web Service"

3. Connect your GitHub account and select your repository

4. Configure your service:
   - **Name**: heart-disease-prediction (or any name you prefer)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free (or paid if you prefer)

5. **Add Environment Variables**:
   - Click "Advanced" → "Add Environment Variable"
   - Add `MONGODB_URI` with your MongoDB connection string from Step 1
   - Add `SECRET_KEY` (you can generate a random string, or Render can auto-generate it)
   - `PORT` is automatically set by Render (no need to add it)

6. Click "Create Web Service"

7. Wait for deployment to complete (usually 2-5 minutes)

### Step 4: Verify Deployment

1. Once deployed, Render will provide a URL like: `https://your-app-name.onrender.com`
2. Visit the URL to test your application
3. Check the logs in Render dashboard if there are any issues

## Important Notes

- **Free Tier Limitations**: Render free tier spins down after 15 minutes of inactivity. First request after spin-down may take 30-60 seconds.
- **Model File**: Make sure `model.pkl` is committed to your repository (it's needed for predictions)
- **Static Files**: All files in the `static/` folder will be served automatically
- **Templates**: All files in the `templates/` folder will be used by Flask

## Troubleshooting

- **MongoDB Connection Issues**: Verify your MongoDB URI is correct and your IP is whitelisted
- **Build Failures**: Check the build logs in Render dashboard
- **Application Errors**: Check the runtime logs in Render dashboard
- **Port Issues**: Render automatically sets the PORT environment variable, so your app should work

## Optional: Using render.yaml

If you prefer, you can use the `render.yaml` file for configuration:
1. Push `render.yaml` to your repository
2. In Render, select "Apply Render YAML" instead of manual configuration
3. Still need to set `MONGODB_URI` environment variable manually

