# justscore

## Environment Setup

1. Copy `.env.template` to `.env`
2. Get the storage account key:
   ```bash
   az storage account keys list --account-name justscoresa --resource-group justscore-rg
   ```
3. Update `.env` with your storage account key

Note: Never commit `.env` files containing secrets to version control.

## Key Rotation
Storage account keys should be rotated periodically:

1. List current keys:
   ```bash
   az storage account keys list --account-name justscoresa --resource-group justscore-rg
   ```

2. Regenerate key2 while using key1:
   ```bash
   az storage account keys renew --account-name justscoresa --key key2
   ```

3. Update applications to use key2

4. Regenerate key1:
   ```bash
   az storage account keys renew --account-name justscoresa --key key1
   ```

This ensures continuous operation while maintaining security.
