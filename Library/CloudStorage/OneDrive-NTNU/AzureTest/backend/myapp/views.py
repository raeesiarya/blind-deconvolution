import jwt
import requests
from jwt.algorithms import RSAAlgorithm
from django.contrib.auth import get_user_model
from rest_framework.decorators import api_view
from rest_framework.response import Response

JWKS_URL = "https://login.microsoftonline.com/common/discovery/v2.0/keys"
CLIENT_ID = "DIN_CLIENT_ID"

def get_keys():
    res = requests.get(JWKS_URL).json()
    return {k["kid"]: RSAAlgorithm.from_jwk(k) for k in res["keys"]}

@api_view(["POST"])
def microsoft_auth(request):
    token = request.data.get("token")
    headers = jwt.get_unverified_header(token)
    key = get_keys().get(headers["kid"])

    if not key:
        return Response({"error": "Key not found"}, status=401)

    try:
        payload = jwt.decode(token, key=key, audience=CLIENT_ID, algorithms=["RS256"])
        email = payload.get("preferred_username")
        User = get_user_model()
        user, _ = User.objects.get_or_create(username=email, defaults={"email": email})
        request.session["user_id"] = user.id
        return Response({"message": "Authenticated", "user": user.username})
    except Exception as e:
        return Response({"error": str(e)}, status=400)
