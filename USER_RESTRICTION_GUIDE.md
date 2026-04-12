# User Access Restriction Guide

## Overview
The BazaarPrime application now supports creating users with restricted date access. A new user `feb_viewer` has been created that can only view data from **February 2026** (Feb 1-28, 2026).

## New User Credentials
- **Username:** `feb_viewer`
- **Password:** Configured in `.streamlit/secrets.toml` (see below)
- **Access:** February 2026 data only (Read-only)

## How It Works

### 1. User Configuration
Two new components were added to `streamlit_app.py`:

#### VALID_USERS Dictionary (Line ~158)
```python
VALID_USERS = {
    "admin": st.secrets.get("user_admin", "admin123"),  # Fallback for development
    "viewer": st.secrets.get("user_viewer", "viewer123"),  # Fallback for development
    "feb_viewer": st.secrets.get("user_feb_viewer", "feb2026secure"),  # Fallback for development
}
```

#### USER_ACCESS_CONTROL Dictionary (Line ~165)
Maps users to their allowed date ranges:
```python
USER_ACCESS_CONTROL = {
    "feb_viewer": {
        "start_date": datetime(2026, 2, 1).date(),
        "end_date": datetime(2026, 2, 28).date(),
        "description": "February 2026 Only",
        "readonly": True,
    }
}
```

### 2. Secrets Configuration
User passwords are now stored in `.streamlit/secrets.toml`:

```toml
# User Authentication
# -------------------
# User passwords for the BazaarPrime dashboard
# Format: user_<username> = "password"
user_admin = "admin123"
user_viewer = "viewer123"
user_feb_viewer = "feb2026secure"
```

**Important:** Never commit the `secrets.toml` file to version control. Add it to your `.gitignore` file.

### 3. Access Control Function
A new function `get_user_date_restriction()` checks if a user has date restrictions:
```python
def get_user_date_restriction(username):
    """Returns (start_date, end_date, is_restricted, description)"""
```

### 4. Enforcement in Sidebar
When a restricted user logs in:
- They see an info box: `📅 Limited Access: February 2026 Only`
- The date period selector is disabled
- A fixed date range is displayed: `02-01-2026 to 02-28-2026`
- Cannot select custom dates or other periods

## Creating Additional Restricted Users

To create more users with date restrictions:

1. **Add password to `.streamlit/secrets.toml`**:
```toml
user_jan_viewer = "secure_password_here"
```

2. **Add to VALID_USERS** (Line ~158):
```python
VALID_USERS = {
    "admin": st.secrets.get("user_admin", "admin123"),
    "viewer": st.secrets.get("user_viewer", "viewer123"),
    "feb_viewer": st.secrets.get("user_feb_viewer", "feb2026secure"),
    "jan_viewer": st.secrets.get("user_jan_viewer", "secure_password_here"),  # New user
}
```

3. **Add to USER_ACCESS_CONTROL** (Line ~165):
```python
USER_ACCESS_CONTROL = {
    "feb_viewer": {
        "start_date": datetime(2026, 2, 1).date(),
        "end_date": datetime(2026, 2, 28).date(),
        "description": "February 2026 Only",
        "readonly": True,
    },
    "jan_viewer": {  # New restriction
        "start_date": datetime(2026, 1, 1).date(),
        "end_date": datetime(2026, 1, 31).date(),
        "description": "January 2026 Only",
        "readonly": True,
    }
}
```

## Example Configurations

### Quarterly Access
```python
"q1_viewer": {
    "start_date": datetime(2026, 1, 1).date(),
    "end_date": datetime(2026, 3, 31).date(),
    "description": "Q1 2026 Only",
    "readonly": True,
}
```

### Specific Range
```python
"march_mid_viewer": {
    "start_date": datetime(2026, 3, 15).date(),
    "end_date": datetime(2026, 3, 31).date(),
    "description": "March 15-31, 2026",
    "readonly": True,
}
```

### Unrestricted User
Users not listed in `USER_ACCESS_CONTROL` have full access to all date ranges:
- Can select any period (Last 7 Days, Last 30 Days, etc.)
- Can use custom date pickers
- Unrestricted data access

## Current Restrictions Applied

✅ **Sidebar Period Selector** - Disabled for restricted users  
✅ **Date Range Display** - Shows fixed range  
✅ **UI Feedback** - Info box shows access level  
✅ **Custom Date Picker** - Disabled for restricted users

## Future Enhancements

The infrastructure is ready for these enhancements:

1. **Database Query-Level Enforcement** - Add restrictions to all SQL queries
2. **Export Restrictions** - Prevent downloading data outside allowed range
3. **Audit Logging** - Log access attempts and data viewed
4. **API Rate Limiting** - Limit API calls per user
5. **Role-Based Access** - Add "analyst", "manager" roles with different permissions

## Testing

To test the restriction:

1. Start the app: `streamlit run streamlit_app.py`
2. Login with:
   - Username: `feb_viewer`
   - Password: Configured in `.streamlit/secrets.toml` (default: `feb2026secure`)
3. Verify:
   - Info box appears showing "Limited Access"
   - Date range shows `02-01-2026 to 02-28-2026`
   - Period selector is replaced with fixed range text
   - No other dates can be selected

## Security Notes

- Passwords are now stored securely in `.streamlit/secrets.toml` (not committed to version control)
- Fallback passwords are provided in code for development environments
- For production, ensure `secrets.toml` is properly secured and not accessible to unauthorized users
- Consider using environment variables or external secret management for production deployments
- All users have access to the same database; filtering is UI-side only
- For complete security, implement database-level access controls

## File Modifications

- **streamlit_app.py** (Lines 156-190, 8900-8940)
  - Added VALID_USERS entry for "feb_viewer"
  - Added USER_ACCESS_CONTROL dictionary
  - Added get_user_date_restriction() function
  - Modified main() sidebar to handle restricted users

## Support

For issues or additional restrictions, contact the development team with:
- Username to create
- Date range required
- Access level (read-only, edit, admin)
