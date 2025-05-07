from django.contrib.auth import authenticate, login, logout, get_user_model
from django.contrib.auth.models import User
from django.contrib.auth.tokens import default_token_generator
from django.contrib.auth.forms import PasswordResetForm, SetPasswordForm
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.contrib.sites.shortcuts import get_current_site
from django.core.mail import EmailMessage
from django.template.loader import render_to_string
from django.conf import settings

from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.authtoken.models import Token 

from .forms import UserRegistrationForm

# Utilisateur modèle
User = get_user_model()


@api_view(['POST'])
@permission_classes([AllowAny])
def login_api(request):
    username = request.data.get('username')
    password = request.data.get('password')

    user = authenticate(request, username=username, password=password)

    if user is not None:
        login(request, user)
        token, created = Token.objects.get_or_create(user=user)
        return Response({
            "success": True,
            "message": "Login successful",
            "token": token.key,
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name
            }
        })
    else:
        return Response({
            "success": False,
            "message": "Invalid credentials"
        }, status=status.HTTP_401_UNAUTHORIZED)


@api_view(['POST'])
@permission_classes([AllowAny])
def register_api(request):
    name_parts = request.data.get('name', '').split(' ')
    form_data = {
        'first_name': name_parts[0],
        'last_name': ' '.join(name_parts[1:]) if len(name_parts) > 1 else '',
        'username': request.data.get('username'),
        'email': request.data.get('email'),
        'password1': request.data.get('password'),
        'password2': request.data.get('password')
    }

    form = UserRegistrationForm(form_data)

    if form.is_valid():
        user = form.save(commit=False)
        user.is_active = False
        user.save()

        # Email d’activation
        current_site = get_current_site(request)
        mail_subject = "Activate your account."
        message = render_to_string('authentication/email_activation/activate_email_message.html', {
            'user': user.username,
            'domain': current_site.domain,
            'uid': urlsafe_base64_encode(force_bytes(user.pk)),
            'token': default_token_generator.make_token(user),
        })

        to_email = form.cleaned_data['email']
        email = EmailMessage(mail_subject, message, to=[to_email])
        email.content_subtype = "html"  # Envoi HTML
        email.send()

        return Response({
            "success": True,
            "message": "Registration successful. Please check your email to activate your account."
        })
    else:
        errors = {field: error_list[0] for field, error_list in form.errors.items()}
        return Response({
            "success": False,
            "message": "Registration failed",
            "errors": errors
        }, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_api(request):
    request.user.auth_token.delete()
    logout(request)
    return Response({
        "success": True,
        "message": "Logout successful"
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def password_reset_request_api(request):
    
    email = request.data.get('email')

    if not email:
        return Response({
            "success": False,
            "message": "Email is required"
        }, status=status.HTTP_400_BAD_REQUEST)

    try:
        # Import traceback for detailed error reporting
        import traceback
        
        # Simplify the query as much as possible
        try:
            # First try with exact match (case sensitive)
            user = User.objects.filter(email=email).first()
        except Exception as db_error:
            print(f"First query attempt failed: {str(db_error)}")
            print(traceback.format_exc())
            # If that fails, try a different approach: manually check emails
            try:
                all_users = User.objects.all()
                user = None
                for u in all_users:
                    if u.email.lower() == email.lower() and u.is_active:
                        user = u
                        break
            except Exception as fallback_error:
                print(f"Fallback query failed: {str(fallback_error)}")
                print(traceback.format_exc())
                raise

        if not user:
            return Response({
                "success": False,
                "message": "User with this email does not exist or is not active."
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Generate token
        token = default_token_generator.make_token(user)
        uid = urlsafe_base64_encode(force_bytes(user.pk))
        current_site = get_current_site(request)
        reset_url = f"http://localhost:3000/auth/password-reset-confirm?uid={uid}&token={token}"

        context = {
            'user': user,
            'uid': uid,
            'token': token,
            'domain': current_site.domain,
            'reset_url': reset_url,
        }

        message = render_to_string('authentication/email_activation/password_reset_email.html', context)

        email_msg = EmailMessage(
            subject="Réinitialisation de votre mot de passe",
            body=message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[email],
        )
        email_msg.content_subtype = "html"
        email_msg.send()

        return Response({
            "success": True,
            "message": "Password reset email has been sent."
        })
    except Exception as e:
        print(f"Password reset error: {str(e)}")
        print(traceback.format_exc())  # Print full traceback for debugging
        return Response({
            "success": False,
            "message": "An error occurred during password reset. Please try again later."
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([AllowAny])
def password_reset_confirm_api(request):
    """
    API endpoint to handle password reset confirmation
    """
    uid = request.data.get('uid')
    token = request.data.get('token')
    new_password1 = request.data.get('new_password1')
    new_password2 = request.data.get('new_password2')

    if not all([uid, token, new_password1, new_password2]):
        return Response({
            "success": False,
            "message": "Missing required fields"
        }, status=status.HTTP_400_BAD_REQUEST)

    try:
        uid_decoded = urlsafe_base64_decode(uid).decode()
        user = User.objects.get(pk=uid_decoded)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user and default_token_generator.check_token(user, token):
        form = SetPasswordForm(user, {
            'new_password1': new_password1,
            'new_password2': new_password2
        })

        if form.is_valid():
            form.save()
            return Response({
                "success": True,
                "message": "Password has been reset successfully"
            })
        else:
            errors = {field: error_list[0] for field, error_list in form.errors.items()}
            return Response({
                "success": False,
                "message": "Password reset failed",
                "errors": errors
            }, status=status.HTTP_400_BAD_REQUEST)
    else:
        return Response({
            "success": False,
            "message": "Invalid reset link"
        }, status=status.HTTP_400_BAD_REQUEST)
