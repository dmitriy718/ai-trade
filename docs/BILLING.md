# Billing & Multi-Tenant (Stripe Integration)

This document describes the Stripe integration and tenant scaffolding for licensing the app on a monthly fee.

## Overview

- **Tenants**: Each paying customer is a tenant. Data (trades, thoughts, performance) is scoped by `tenant_id`.
- **Stripe**: Create customers and monthly subscriptions via Checkout; webhooks update tenant status (`active`, `past_due`, `canceled`).
- **Default tenant**: When not using billing, the app uses `tenant_id = "default"`. All existing data stays under default.

## Config

In `config/config.yaml`:

```yaml
billing:
  stripe:
    enabled: true
    secret_key: ""       # or STRIPE_SECRET_KEY in .env
    webhook_secret: ""   # or STRIPE_WEBHOOK_SECRET
    price_id: ""         # Stripe Price ID for monthly plan (price_...)
    currency: usd
  tenant:
    default_tenant_id: default
```

## Stripe Setup

1. **Stripe Dashboard**: Create a Product and a recurring Price (monthly). Copy the Price ID (`price_...`).
2. **Webhook**: Add endpoint `https://your-domain/api/v1/billing/webhook`, subscribe to:
   - `checkout.session.completed`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
   - `invoice.paid`
   - `invoice.payment_failed`
   Copy the webhook signing secret (`whsec_...`).
3. **Keys**: Use test keys (`sk_test_...`, `whsec_...`) for development; live keys for production.

## API

- **POST /api/v1/billing/checkout** (requires X-API-Key)  
  Body: `{ "tenant_id": "acme", "success_url": "https://...", "cancel_url": "https://...", "customer_email": "optional" }`  
  Returns `{ "url": "https://checkout.stripe.com/...", "session_id": "..." }`. Redirect the user to `url` to pay.

- **POST /api/v1/billing/webhook**  
  Called by Stripe. No auth; verified by `Stripe-Signature` header. Updates tenant status in DB.

- **GET /api/v1/tenants/{tenant_id}**  
  Returns tenant (id, name, stripe_customer_id, stripe_subscription_id, status).

## Database

- **tenants**: id, name, stripe_customer_id, stripe_subscription_id, status, created_at, updated_at.
- **tenant_api_keys**: api_key_hash → tenant_id (for future API-key → tenant resolution).
- **tenant_id** column added to: trades, thought_log, signals, ml_features, daily_summary. Default `'default'`.

Migrations run on DB init; existing rows get `tenant_id = 'default'`.

## Enabling a New Client

1. Create a tenant (e.g. via admin or API): `upsert_tenant(tenant_id, name, ...)`.
2. Call **POST /api/v1/billing/checkout** with that `tenant_id`, success_url, cancel_url.
3. User completes Stripe Checkout. On `checkout.session.completed`, the webhook stores `stripe_customer_id` and `stripe_subscription_id` on the tenant and sets status `active`.
4. Gate dashboard/API by tenant status (e.g. only allow access when `tenant.status == 'active'`).

## Future

- Resolve tenant from **X-Tenant-ID** header or API key (tenant_api_keys) in middleware; pass tenant_id to DB calls.
- Per-tenant config (pairs, strategies) in DB or tenant_config table.
- Customer Portal link: use `create_billing_portal_session(customer_id, return_url)` for “Manage subscription”.
