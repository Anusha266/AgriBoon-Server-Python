from django.db import models
from django.contrib.auth.hashers import make_password, check_password
import uuid


class User(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=255)
    phone = models.CharField(max_length=20)
    # Profile fields
    profilePic = models.ImageField(upload_to="profile_pics/", blank=True, null=True)
    state = models.CharField(max_length=100, blank=True, default="")
    mandal = models.CharField(max_length=100, blank=True, default="")
    district = models.CharField(max_length=100, blank=True, default="")
    village = models.CharField(max_length=100, blank=True, default="")
    address = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    def set_password(self, raw_password):
        self.password = make_password(raw_password)

    def check_password(self, raw_password):
        return check_password(raw_password, self.password)

    def to_dict(self):
        if self.profilePic:
            pic_url = f"http://localhost:8000{self.profilePic.url}" if not self.profilePic.url.startswith("http") else self.profilePic.url
        else:
            pic_url = ""
        return {
            "_id": str(self.id),
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "profilePic": pic_url,
            "state": self.state,
            "mandal": self.mandal,
            "district": self.district,
            "village": self.village,
            "address": self.address,
        }

    def __str__(self):
        return self.name


class Product(models.Model):
    STATUS_CHOICES = [
        ("available", "Available"),
        ("pending", "Pending"),
        ("accepted", "Accepted"),
        ("success", "Success"),
        ("failed", "Failed"),
        ("deny", "Deny"),
    ]
    name = models.CharField(max_length=255)
    image = models.ImageField(upload_to="products/")
    quality = models.CharField(max_length=50, blank=True, default="")
    min_price = models.FloatField(default=0)
    max_price = models.FloatField(default=0)
    uploaded_on = models.CharField(max_length=50, blank=True, default="")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="available")
    cart = models.BooleanField(default=False)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="products")
    createdOn = models.DateTimeField(auto_now_add=True)

    def to_dict(self, include_owner=False, request=None):
        if self.image:
            image_url = f"http://localhost:8000{self.image.url}" if not self.image.url.startswith("http") else self.image.url
        else:
            image_url = ""
        data = {
            "_id": str(self.id),
            "id": str(self.id),
            "name": self.name,
            "image": image_url,
            "quality": self.quality,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "uploaded_on": self.uploaded_on,
            "status": self.status,
            "cart": self.cart,
            "createdOn": self.createdOn.isoformat() if self.createdOn else "",
        }
        if include_owner:
            data["owner_details"] = self.owner.to_dict()
        return data

    def __str__(self):
        return f"{self.name} by {self.owner.name}"


class Transaction(models.Model):
    buyer = models.ForeignKey(User, on_delete=models.CASCADE, related_name="bought_transactions")
    seller = models.ForeignKey(User, on_delete=models.CASCADE, related_name="sold_transactions")
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="transactions")
    isCompleted = models.BooleanField(default=False)
    isOTP_active = models.BooleanField(default=False)
    sellerOTP = models.CharField(max_length=10, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    def to_dict(self):
        return {
            "_id": str(self.id),
            "buyer": self.buyer.to_dict(),
            "seller": self.seller.to_dict(),
            "product": self.product.to_dict(include_owner=True),
            "isCompleted": self.isCompleted,
            "isOTP_active": self.isOTP_active,
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }

    def __str__(self):
        return f"Transaction {self.id}: {self.buyer.name} -> {self.seller.name}"
