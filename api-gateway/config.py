import os
from dotenv import load_dotenv

# .env dosyasından çevre değişkenlerini yükle
load_dotenv()

# API servisleri için yapılandırma
RETURN_RISK_API_URL = os.getenv("RETURN_RISK_API_URL", "http://localhost:8001")
NEXT_ORDER_API_URL = os.getenv("NEXT_ORDER_API_URL", "http://localhost:8002")
NEW_PRODUCT_API_URL = os.getenv("NEW_PRODUCT_API_URL", "http://localhost:8003")

# API Gateway yapılandırması
API_GATEWAY_HOST = os.getenv("API_GATEWAY_HOST", "0.0.0.0")
API_GATEWAY_PORT = int(os.getenv("API_GATEWAY_PORT", "8080")) 