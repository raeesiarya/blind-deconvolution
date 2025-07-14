## Project
1. Django backend  
2. Vue + Vite + TypeScript frontend

---

### Relevant files for Azure

- **Backend**
    - [`backend/myapp/urls.py`](backend/myapp/urls.py)
    - [`backend/myapp/views.py`](backend/myapp/views.py)
    - [`backend/backend/settings.py`](backend/backend/settings.py)
    - [`backend/backend/urls.py`](backend/backend/urls.py)
- **Frontend**
    - [`frontend/src/App.vue`](frontend/src/App.vue)
    - [`frontend/src/auth.ts`](frontend/src/auth.ts)

### Azure setup
1. Create resource group in Azure
2. Create tenant in resource group
2. Create app in Azure
3. Create Azure Entra ID in Azure (external in this project since it's a personal test project)

---

### Screenshots

#### Azure resource group
![Azure resource group](screenshots/Skjermbilde%202025-07-14%20kl.%2018.52.03.png)

#### Azure external tenant
![Azure external tenant](screenshots/Skjermbilde%202025-07-14%20kl.%2018.52.11.png)

#### Azure app
![Azure app](screenshots/Skjermbilde%202025-07-14%20kl.%2018.53.51.png)

#### Azure app API permissions (important for login!)
![Azure app API permissions (important for login!)](screenshots/Skjermbilde%202025-07-14%20kl.%2018.54.10.png)