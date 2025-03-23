# Setting Up Azure Credentials for GitHub Actions

This guide will help you set up the necessary Azure credentials for the GitHub Actions workflows.

## Detailed Setup Instructions

### 1. Install Azure CLI

If you haven't already, [install the Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli).

### 2. Login to Azure

```bash
az login
```

Follow the instructions to complete the login process.

### 3. Verify your subscription

```bash
az account show
```

If you have multiple subscriptions, set the appropriate one:

```bash
az account set --subscription 1
```

### 4. Create a Service Principal

Run this command to create a service principal with Contributor permissions scoped to your resource group:

```bash
az ad sp create-for-rbac \
  --name "JustScoreGitHubActions" \
  --role Contributor \
  --scopes /subscriptions/1/resourceGroups/justscore-rg \
  --sdk-auth
```

This will output JSON similar to:

```json
{
  "clientId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "clientSecret": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "subscriptionId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "tenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",
  "resourceManagerEndpointUrl": "https://management.azure.com/",
  "activeDirectoryGraphResourceId": "https://graph.windows.net/",
  "sqlManagementEndpointUrl": "https://management.core.windows.net:8443/",
  "galleryEndpointUrl": "https://gallery.azure.com/",
  "managementEndpointUrl": "https://management.core.windows.net/"
}
```

### 5. Add the credentials to GitHub Secrets

1. Go to your GitHub repository
2. Navigate to Settings > Secrets and Variables > Actions
3. Click "New repository secret"
4. Name: `AZURE_CREDENTIALS`
5. Value: Paste the entire JSON output from step 4
6. Click "Add secret"

### 6. Get Azure Container Registry credentials

```bash
az acr credential show --name justscoreacr
```

This will output the username and passwords for your ACR.

### 7. Add ACR credentials to GitHub Secrets

Add two more secrets:

1. `ACR_USERNAME`: The username from step 6
2. `ACR_PASSWORD`: One of the passwords from step 6

## Troubleshooting

### Common Errors

#### "Login failed with Error: Using auth-type: SERVICE_PRINCIPAL"

This usually means the `AZURE_CREDENTIALS` secret is missing or malformed. Make sure:
- The secret is called exactly `AZURE_CREDENTIALS`
- The value is the complete JSON output (including curly braces)
- There are no extra spaces or line breaks

#### "unauthorized: authentication required"

This occurs when ACR credentials are incorrect. Verify:
- `ACR_USERNAME` and `ACR_PASSWORD` are set correctly
- The service principal has permissions to access the container registry

### Checking Permissions

To verify the service principal has the right permissions:

```bash
az role assignment list --assignee "<client-id-from-above>"
```

### Revoking Access

If you need to revoke access for security reasons:

```bash
az ad sp delete --id "<client-id-from-above>"
```

## Additional Resources

- [Azure Login Action](https://github.com/Azure/login)
- [Azure WebApp Deploy Action](https://github.com/Azure/webapps-deploy)
- [Azure RBAC Documentation](https://docs.microsoft.com/en-us/azure/role-based-access-control/overview) 