from src.services.churn_service import ChurnService

service = ChurnService()

sample = {
    "age": 25,
    "last_interaction": 5,
    "num_sessions": 25,
    "purchase_frequency": 3,
    "support_tickets": 1,
    "tenure": 24
}

print(service.predict(sample))