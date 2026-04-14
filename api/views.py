import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
from PIL import Image
import numpy as np
import razorpay
from google import genai
import requests as http_requests
from decouple import config

from .models import User, Product, Transaction
from .auth_utils import generate_token, get_user_from_token

# Razorpay client
razorpay_client = razorpay.Client(auth=(config('RAZORPAY_KEY_ID'), config('RAZORPAY_KEY_SECRET')))

# Gemini client
CROP_ADVISOR_SYSTEM_PROMPT = """You are AgriBoon AI, an expert agricultural advisor for Indian farmers. You help with:
- Crop selection and planting schedules based on region and season
- Soil health, fertilizers, and pest management
- Weather-based farming advice
- Market pricing and when to sell
- Organic farming techniques
- Government schemes for farmers

Keep responses concise (2-3 paragraphs max), practical, and easy to understand.
Use simple language as many users are farmers with basic education.
When relevant, mention specific Indian states, crops, and local practices.
If asked something unrelated to agriculture, politely redirect to farming topics."""

gemini_client = genai.Client(api_key=config('GEMINI_API_KEY'))

quality_mapping = {'low': 0.2, 'medium': 0.50, 'high': 1}


# ==================== AUTH ====================

class SignupView(APIView):
    def post(self, request):
        name = request.data.get("name")
        email = request.data.get("email")
        password = request.data.get("password")
        phone = request.data.get("phone")

        if not all([name, email, password, phone]):
            return Response({"message": "All fields are required."}, status=status.HTTP_400_BAD_REQUEST)

        if User.objects.filter(email=email).exists():
            return Response({"message": "Email already registered."}, status=status.HTTP_400_BAD_REQUEST)

        user = User(name=name, email=email, phone=phone)
        user.set_password(password)
        user.save()

        token = generate_token(user)
        return Response({"token": token, "message": "Signup successful"}, status=status.HTTP_201_CREATED)


class LoginView(APIView):
    def post(self, request):
        email = request.data.get("email")
        password = request.data.get("password")

        if not email or not password:
            return Response({"message": "Email and password are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response({"message": "Invalid credentials."}, status=status.HTTP_401_UNAUTHORIZED)

        if not user.check_password(password):
            return Response({"message": "Invalid credentials."}, status=status.HTTP_401_UNAUTHORIZED)

        token = generate_token(user)
        return Response({"token": token, "message": "Login successful"}, status=status.HTTP_200_OK)


class CurrentUserView(APIView):
    def get(self, request):
        user = get_user_from_token(request)
        if not user:
            return Response({"message": "Unauthorized"}, status=status.HTTP_401_UNAUTHORIZED)
        return Response({"data": user.to_dict()}, status=status.HTTP_200_OK)


# ==================== PROFILE ====================

class ProfileCreateView(APIView):
    def post(self, request):
        user = get_user_from_token(request)
        if not user:
            return Response({"message": "Unauthorized"}, status=status.HTTP_401_UNAUTHORIZED)

        user.state = request.data.get("state", user.state)
        user.mandal = request.data.get("mandal", user.mandal)
        user.district = request.data.get("district", user.district)
        user.village = request.data.get("village", user.village)
        user.address = request.data.get("address", user.address)

        if "profilePic" in request.FILES:
            user.profilePic = request.FILES["profilePic"]

        user.save()
        return Response({"message": "Profile updated successfully", "data": user.to_dict()}, status=status.HTTP_200_OK)


# ==================== PRODUCTS ====================

class ProductUploadView(APIView):
    def post(self, request):
        user = get_user_from_token(request)
        if not user:
            return Response({"message": "Unauthorized"}, status=status.HTTP_401_UNAUTHORIZED)

        image = request.FILES.get("image")
        if not image:
            return Response({"message": "Image is required."}, status=status.HTTP_400_BAD_REQUEST)

        product = Product.objects.create(
            name=request.data.get("name", ""),
            image=image,
            quality=request.data.get("quality", ""),
            min_price=float(request.data.get("min_price", 0)),
            max_price=float(request.data.get("max_price", 0)),
            uploaded_on=request.data.get("uploaded_on", ""),
            owner=user,
        )
        return Response({"message": "Product uploaded", "data": product.to_dict()}, status=status.HTTP_201_CREATED)


class AllProductsView(APIView):
    def get(self, request):
        products = Product.objects.filter(status="available").order_by("-createdOn")
        return Response({"data": [p.to_dict(include_owner=True) for p in products]}, status=status.HTTP_200_OK)


class ProductsByNameView(APIView):
    def get(self, request):
        name = request.GET.get("name", "")
        products = Product.objects.filter(name__icontains=name, status="available").order_by("-createdOn")
        return Response({"data": [p.to_dict(include_owner=True) for p in products]}, status=status.HTTP_200_OK)


class ProductByIdView(APIView):
    def get(self, request, product_id):
        try:
            product = Product.objects.get(id=product_id)
            return Response({"data": [product.to_dict(include_owner=True)]}, status=status.HTTP_200_OK)
        except Product.DoesNotExist:
            return Response({"message": "Product not found."}, status=status.HTTP_404_NOT_FOUND)


class ProductUpdateView(APIView):
    def patch(self, request, product_id):
        try:
            product = Product.objects.get(id=product_id)
        except Product.DoesNotExist:
            return Response({"message": "Product not found."}, status=status.HTTP_404_NOT_FOUND)

        if "status" in request.data:
            product.status = request.data["status"]
        if "cart" in request.data:
            product.cart = request.data["cart"]

        product.save()
        return Response({"message": "Product updated", "data": product.to_dict()}, status=status.HTTP_200_OK)


class CartProductsView(APIView):
    def get(self, request):
        products = Product.objects.filter(cart=True).order_by("-createdOn")
        return Response({"data": [p.to_dict() for p in products]}, status=status.HTTP_200_OK)


# ==================== TRANSACTIONS ====================

class CreateTransactionView(APIView):
    def post(self, request):
        user = get_user_from_token(request)
        if not user:
            return Response({"message": "Unauthorized"}, status=status.HTTP_401_UNAUTHORIZED)

        product_id = request.data.get("product")
        seller_id = request.data.get("user")

        try:
            product = Product.objects.get(id=product_id)
            seller = User.objects.get(id=seller_id)
        except (Product.DoesNotExist, User.DoesNotExist):
            return Response({"message": "Product or seller not found."}, status=status.HTTP_404_NOT_FOUND)

        transaction = Transaction.objects.create(buyer=user, seller=seller, product=product)
        return Response({"message": "Transaction created", "data": transaction.to_dict()}, status=status.HTTP_201_CREATED)


class TransactionsByProductStatusView(APIView):
    def get(self, request, product_status):
        user = get_user_from_token(request)
        if not user:
            return Response({"message": "Unauthorized"}, status=status.HTTP_401_UNAUTHORIZED)

        transactions = Transaction.objects.filter(product__status=product_status).order_by("-created_at")
        data = [t.to_dict() for t in transactions]
        return Response({"data": data}, status=status.HTTP_200_OK)


class BuyerDataView(APIView):
    def post(self, request):
        user = get_user_from_token(request)
        if not user:
            return Response({"message": "Unauthorized"}, status=status.HTTP_401_UNAUTHORIZED)

        user_id = request.data.get("userId")
        product_id = request.data.get("productId")
        req_status = request.data.get("status")

        transactions = Transaction.objects.filter(
            buyer_id=user_id, product_id=product_id, product__status=req_status
        )
        data = [t.to_dict() for t in transactions]
        return Response({"data": data}, status=status.HTTP_200_OK)


class SellerDataView(APIView):
    def post(self, request):
        user = get_user_from_token(request)
        if not user:
            return Response({"message": "Unauthorized"}, status=status.HTTP_401_UNAUTHORIZED)

        owner_id = request.data.get("ownerId")
        product_id = request.data.get("productId")
        req_status = request.data.get("status")

        transactions = Transaction.objects.filter(
            seller_id=owner_id, product_id=product_id, product__status=req_status
        )
        data = [t.to_dict() for t in transactions]
        return Response({"data": data}, status=status.HTTP_200_OK)


class UpdateTransactionView(APIView):
    def patch(self, request, transaction_id):
        try:
            transaction = Transaction.objects.get(id=transaction_id)
        except Transaction.DoesNotExist:
            return Response({"message": "Transaction not found."}, status=status.HTTP_404_NOT_FOUND)

        if "isCompleted" in request.data:
            transaction.isCompleted = request.data["isCompleted"]
        if "isOTP_active" in request.data:
            transaction.isOTP_active = request.data["isOTP_active"]
        if "sellerOTP" in request.data:
            transaction.sellerOTP = str(request.data["sellerOTP"])

        transaction.save()
        return Response({"message": "Transaction updated", "data": transaction.to_dict()}, status=status.HTTP_200_OK)


class GetTransactionByIdView(APIView):
    def get(self, request, transaction_id):
        try:
            transaction = Transaction.objects.get(id=transaction_id)
            return Response({"data": transaction.to_dict()}, status=status.HTTP_200_OK)
        except Transaction.DoesNotExist:
            return Response({"message": "Transaction not found."}, status=status.HTTP_404_NOT_FOUND)


class TransactionByProductDetailsView(APIView):
    def post(self, request):
        user = get_user_from_token(request)
        if not user:
            return Response({"message": "Unauthorized"}, status=status.HTTP_401_UNAUTHORIZED)

        product_id = request.data.get("productId")
        try:
            transaction = Transaction.objects.filter(product_id=product_id).latest("created_at")
            return Response({"data": transaction.to_dict()}, status=status.HTTP_200_OK)
        except Transaction.DoesNotExist:
            return Response({"message": "Transaction not found."}, status=status.HTTP_404_NOT_FOUND)


class CompareOTPView(APIView):
    def post(self, request):
        transaction_id = request.data.get("id")
        otp = request.data.get("otp")

        try:
            transaction = Transaction.objects.get(id=transaction_id)
            is_equal = str(transaction.sellerOTP) == str(otp)
            return Response({"data": {"isEqual": is_equal}}, status=status.HTTP_200_OK)
        except Transaction.DoesNotExist:
            return Response({"message": "Transaction not found."}, status=status.HTTP_404_NOT_FOUND)


# ==================== AI CHATBOT ====================


class WeatherView(APIView):
    def get(self, request):
        # Use city from query param, or from user's profile location
        city = request.GET.get("city", "")
        if not city:
            user = get_user_from_token(request)
            if user and user.state:
                city = user.state
            elif user and user.district:
                city = user.district
            else:
                city = "Hyderabad"

        api_key = config('OPENWEATHER_API_KEY', default='')
        if not api_key or api_key == 'your_openweathermap_api_key_here':
            return Response({"data": None, "message": "Weather service is not configured yet."}, status=status.HTTP_200_OK)

        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
            resp = http_requests.get(url, timeout=5)
            if resp.status_code != 200:
                return Response({"data": None, "message": "Unable to fetch weather data."}, status=status.HTTP_200_OK)
            w = resp.json()
            data = {
                "city": w.get("name", city),
                "temp": round(w["main"]["temp"]),
                "feels_like": round(w["main"]["feels_like"]),
                "humidity": w["main"]["humidity"],
                "description": w["weather"][0]["description"].title() if w.get("weather") else "",
                "icon": w["weather"][0]["icon"] if w.get("weather") else "",
                "wind_speed": round(w.get("wind", {}).get("speed", 0) * 3.6),  # m/s to km/h
            }
            return Response({"data": data}, status=status.HTTP_200_OK)
        except Exception:
            return Response({"data": None, "message": "Weather service is temporarily unavailable."}, status=status.HTTP_200_OK)


class PriceTrendsView(APIView):
    def get(self, request):
        product_name = request.GET.get("product", "groundnut").lower()
        try:
            data = pd.read_csv(f'AgriBoon-Backend-Django/data/{product_name}_price_data_2023.csv')
            data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
            # Sample to ~52 weekly points for chart performance
            data = data.iloc[::7].reset_index(drop=True)
            records = []
            for _, row in data.iterrows():
                records.append({
                    "date": row['date'].strftime('%b %d'),
                    "min_price": round(row['min_price']),
                    "modal_price": round(row['modal_price']),
                    "max_price": round(row['max_price']),
                })
            return Response({"data": records}, status=status.HTTP_200_OK)
        except FileNotFoundError:
            return Response({"data": [], "message": "No data available for this product."}, status=status.HTTP_200_OK)
        except Exception:
            return Response({"data": [], "message": "Error loading price data."}, status=status.HTTP_200_OK)


class ChatbotView(APIView):
    def post(self, request):
        user_message = request.data.get("message", "").strip()
        if not user_message:
            return Response({"message": "Please enter a message."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            response = gemini_client.models.generate_content(
                model='gemini-2.0-flash',
                config={'system_instruction': CROP_ADVISOR_SYSTEM_PROMPT, 'max_output_tokens': 512},
                contents=user_message,
            )
            reply = response.text
            return Response({"data": {"reply": reply}}, status=status.HTTP_200_OK)
        except Exception:
            return Response({"data": {"reply": "Unable to reach the AI advisor at the moment. Please try again later."}}, status=status.HTTP_200_OK)


# ==================== RAZORPAY PAYMENT ====================

class CreateRazorpayOrderView(APIView):
    def post(self, request):
        amount = request.data.get("amount")
        transaction_id = request.data.get("transactionId")

        if not amount:
            return Response({"message": "Amount is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            order_data = {
                "amount": int(float(amount) * 100),  # Razorpay expects paise
                "currency": "INR",
                "receipt": f"txn_{transaction_id}",
                "payment_capture": 1,  # Auto-capture
            }
            order = razorpay_client.order.create(data=order_data)
            return Response({"data": order}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"message": f"Error creating order: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ==================== ML PRICE PREDICTION ====================

class PricePredictionAPIView(APIView):
    def post(self, request, *args, **kwargs):
        image = request.FILES.get('image')
        inflation_rate = request.data.get('inflation_rate')
        product_name = request.data.get('product_name')

        if not image or inflation_rate is None or product_name is None:
            return Response({"error": "Missing required fields."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            min_price_model = joblib.load('AgriBoon-Backend-Django/ml_models/dynamic_min_price_model.pkl')
            modal_price_model = joblib.load('AgriBoon-Backend-Django/ml_models/dynamic_modal_price_model.pkl')
            max_price_model = joblib.load('AgriBoon-Backend-Django/ml_models/dynamic_max_price_model.pkl')
        except Exception as e:
            return Response({"error": "Error loading models."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        quality = self.predict_quality(image)
        if quality is None:
            return Response({"error": "Unable to predict quality."}, status=status.HTTP_400_BAD_REQUEST)

        today = pd.to_datetime('today').strftime('%Y-%m-%d')

        price_x_min = self.get_price_data(today, 'min_price', product_name)
        price_x_modal = self.get_price_data(today, 'modal_price', product_name)
        price_x_max = self.get_price_data(today, 'max_price', product_name)

        for price_type, price_x, model in [
            ('min', price_x_min, min_price_model),
            ('modal', price_x_modal, modal_price_model),
            ('max', price_x_max, max_price_model),
        ]:
            pass

        new_data_min = {'date': [today], 'price_x': [price_x_min], 'quality': [quality_mapping.get(quality, 0.5)], 'inflation_rate': [inflation_rate]}
        new_data_modal = {'date': [today], 'price_x': [price_x_modal], 'quality': [quality_mapping.get(quality, 0.5)], 'inflation_rate': [inflation_rate]}
        new_data_max = {'date': [today], 'price_x': [price_x_max], 'quality': [quality_mapping.get(quality, 0.5)], 'inflation_rate': [inflation_rate]}

        dfs = []
        for d in [new_data_min, new_data_modal, new_data_max]:
            df = pd.DataFrame(d)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['day_of_year'] = df['date'].dt.dayofyear
            dfs.append(df)

        min_price_pred = min_price_model.predict(dfs[0][['day_of_year', 'quality', 'price_x', 'inflation_rate']])
        modal_price_pred = modal_price_model.predict(dfs[1][['day_of_year', 'quality', 'price_x', 'inflation_rate']])
        max_price_pred = max_price_model.predict(dfs[2][['day_of_year', 'quality', 'price_x', 'inflation_rate']])

        return Response({
            "data": {
                "min_price": min_price_pred[0],
                "modal_price": modal_price_pred[0],
                "max_price": max_price_pred[0],
                "Quality": quality,
            }
        }, status=status.HTTP_200_OK)

    def predict_quality(self, image):
        try:
            img = Image.open(image).convert('RGB').resize((224, 224))
            pixels = np.array(img, dtype=np.float64)
            brightness = np.mean(pixels)
            saturation = np.mean(np.std(pixels, axis=2))
            contrast = np.std(pixels)
            score = (brightness / 255) * 0.4 + (saturation / 128) * 0.3 + (contrast / 128) * 0.3
            if score > 0.55:
                return 'high'
            elif score > 0.35:
                return 'medium'
            return 'low'
        except Exception:
            return None

    def get_price_data(self, date, price_type, product_name):
        try:
            target_date = pd.to_datetime(date)
            data = pd.read_csv(f'AgriBoon-Backend-Django/data/{product_name.lower()}_price_data_2023.csv')
            data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
            data = data.sort_values(by='date')
            exact = data[data['date'] == target_date]
            if not exact.empty:
                return exact.iloc[0][price_type]
            before = data[data['date'] < target_date].iloc[-1:]
            if not before.empty:
                return before.iloc[0][price_type]
            after = data[data['date'] > target_date].iloc[:1]
            if not after.empty:
                return after.iloc[0][price_type]
            return data.iloc[0][price_type]
        except Exception:
            return None
