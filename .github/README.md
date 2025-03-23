# GitHub Actions for JustScore

This directory contains GitHub Actions workflows for building, deploying, and promoting the JustScore application.

## Available Workflows

1. **deploy.yml** - Builds and deploys the application when code is pushed to main
2. **promote-to-production.yml** - Promotes an existing container image to production

## Setup Instructions

### 1. Configure GitHub Secrets

You need to set up the following secrets in your GitHub repository:

- `AZURE_CREDENTIALS`: Service principal credentials for Azure
  ```json
  {
    "clientId": "<client-id>",
    "clientSecret": "<client-secret>",
    "subscriptionId": "<subscription-id>",
    "tenantId": "<tenant-id>"
  }
  ```

- `ACR_USERNAME`: Username for Azure Container Registry
- `ACR_PASSWORD`: Password for Azure Container Registry

### 2. Create Service Principal

Run the following Azure CLI commands to create a service principal:

```bash
# Login to Azure
az login

# Create a service principal with Contributor role scoped to the resource group
az ad sp create-for-rbac --name "JustScoreGitHubActions" \
  --role Contributor \
  --scopes /subscriptions/<subscription-id>/resourceGroups/justscore-rg \
  --sdk-auth
```

Copy the JSON output and add it as the `AZURE_CREDENTIALS` secret in GitHub.

### 3. Get ACR Credentials

```bash
# Get ACR credentials
az acr credential show --name justscoreacr
```

Use the username and password values for the `ACR_USERNAME` and `ACR_PASSWORD` secrets.

## Using the Workflows

### Deploy Application

The `deploy.yml` workflow runs automatically when code is pushed to the main branch.

### Promote to Production

To promote an existing container image to production:

1. Go to the "Actions" tab in your GitHub repository
2. Select "Promote to Production" workflow
3. Click "Run workflow"
4. Enter the image tag to promote (defaults to "latest")
5. Click "Run workflow" 