from django.urls import path
from .views import (
    SignupView, LoginView, CurrentUserView,
    ProfileCreateView,
    ProductUploadView, AllProductsView, ProductsByNameView,
    ProductByIdView, ProductUpdateView, CartProductsView,
    CreateTransactionView, TransactionsByProductStatusView,
    BuyerDataView, SellerDataView, UpdateTransactionView,
    GetTransactionByIdView, TransactionByProductDetailsView,
    CompareOTPView, WeatherView, PriceTrendsView, ChatbotView, CreateRazorpayOrderView, PricePredictionAPIView,
)

app_name = 'api'

urlpatterns = [
    # Auth
    path('auth/signup', SignupView.as_view()),
    path('auth/login', LoginView.as_view()),
    path('auth/current-user', CurrentUserView.as_view()),

    # Profile
    path('profile/create', ProfileCreateView.as_view()),

    # Products
    path('products/upload', ProductUploadView.as_view()),
    path('products/all', AllProductsView.as_view()),
    path('products/get-by-name', ProductsByNameView.as_view()),
    path('products/get/<int:product_id>', ProductByIdView.as_view()),
    path('products/update/<int:product_id>', ProductUpdateView.as_view()),
    path('products/cart', CartProductsView.as_view()),

    # Transactions
    path('transactions/create', CreateTransactionView.as_view()),
    path('transactions/status/<str:product_status>', TransactionsByProductStatusView.as_view()),
    path('transactions/buyer-data', BuyerDataView.as_view()),
    path('transactions/seller-data', SellerDataView.as_view()),
    path('transactions/update/<int:transaction_id>', UpdateTransactionView.as_view()),
    path('transactions/get/<int:transaction_id>', GetTransactionByIdView.as_view()),
    path('transactions/get-by-product', TransactionByProductDetailsView.as_view()),
    path('transactions/compare-otp', CompareOTPView.as_view()),

    # Weather
    path('weather', WeatherView.as_view()),

    # Price Trends
    path('price-trends', PriceTrendsView.as_view()),

    # AI Chatbot
    path('chatbot', ChatbotView.as_view()),

    # Razorpay Payment
    path('payments/create-order', CreateRazorpayOrderView.as_view()),

    # ML Price Prediction
    path('fetch_price/', PricePredictionAPIView.as_view()),
]
