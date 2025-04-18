#!/bin/bash

# Login to Azure
az login

# Set your Azure subscription
# az account set --subscription "your-subscription-id"

# Create a resource group (if not exists)
az group create --name mdst-app-rg --location eastus

# Create an Azure Container Registry (if not exists)
az acr create --resource-group mdst-app-rg --name mdstappregistry --sku Basic

# Login to the container registry
az acr login --name mdstappregistry

# Build and push the Docker image
docker build -t mdstappregistry.azurecr.io/mdst-app:latest .
docker push mdstappregistry.azurecr.io/mdst-app:latest

# Create an App Service plan
az appservice plan create --name mdst-app-plan --resource-group mdst-app-rg --sku B1 --is-linux

# Create the web app
az webapp create --resource-group mdst-app-rg --plan mdst-app-plan --name mdst-app --deployment-container-image-name mdstappregistry.azurecr.io/mdst-app:latest

# Configure the web app to use the container registry
az webapp config container set --name mdst-app --resource-group mdst-app-rg --docker-custom-image-name mdstappregistry.azurecr.io/mdst-app:latest --docker-registry-server-url https://mdstappregistry.azurecr.io

echo "Deployment completed! Your app is available at: https://mdst-app.azurewebsites.net"