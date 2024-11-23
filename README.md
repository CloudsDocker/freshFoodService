# Keep-It-Fresh
Application Iteration 1


# To run the server locally

## Fistly setup project level virutal environmen
```shell
python -m venv monash_fresh_env
source monash_fresh_env/bin/activate
pip install -r requirements.txt
```
Then setup local environment and run the server

```shell

export PYTHONPATH=$PYTHONPATH:/Users/todd.zhang/dev/ws/monash/freshFoodService
export DJANGO_SETTINGS_MODULE=KeepItFresh.settings
python manage.py runserver
```