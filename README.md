# constellcalculator

Weird data twitter bot [@constellcalculator](https://twitter.com/constellcalc), using AWS lambda and [Zappa](https://github.com/Miserlou/Zappa)

# configuration

1. clone the repo: `git clone git@github.com:rhydomako/constellcalculator.git`
2. create and activate an virturalenv: `virtualenv env; . env/bin/activate`
3. install python requirements: `pip install -r requirements.txt`
4. configure initial zappa settings: `zappa init`, and follow the config instructions
5. add twitter creds and scheduling info to `zappa_settings.json`:
```
{
    "prod": {
        "app_function": "my_app.app",
        "aws_region": "us-east-1",
        "profile_name": "default",
        "s3_bucket": "XXXXX"
        "environment_variables": {
            "TWITTER_TOKEN": "XXXXX",
            "TWITTER_TOKENSECRET": "XXXXX",
            "TWITTER_CONSUMERTOKEN": "XXXXX",
            "TWITTER_CONSUMERSECRET": "XXXX"
        }
        "events": [{
           "function": "my_app.post", // The function to execute
           "expression": "rate(90 minutes)" // When to execute it (in cron or rate format)
        }],
    }
}
```
6. Deploy: `zappa deploy prod`
