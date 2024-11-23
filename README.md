# freshFoodService
The backend API service for freshFood


# develop

## How to safely handle Azure token

### First, remove the sensitive key from your settings.py. Replace the actual key with an environment variable:
```python
# In settings.py
# Instead of having the key directly like:
# AZURE_STORAGE_KEY = "your_actual_key_here"

# Use environment variable instead:
AZURE_STORAGE_KEY = os.environ.get('AZURE_STORAGE_KEY')
```
### Create a .env file to store your actual credentials locally:

```shell
echo "AZURE_STORAGE_KEY=your_actual_key_here" >> .env
```
### Add .env to your .gitignore:
```shell
echo ".env" >> .gitignore
```