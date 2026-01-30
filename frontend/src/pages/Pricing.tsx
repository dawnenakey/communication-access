import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { supabase } from '@/lib/supabase';
import {
  Check, X, ArrowRight, Sparkles, Shield, Zap, Crown,
  Hand, ArrowLeft, Loader2, CreditCard, Monitor, Video
} from 'lucide-react';

interface Plan {
  id: string;
  name: string;
  description: string;
  price: { monthly: number; annual: number };
  icon: React.ReactNode;
  color: string;
  borderColor: string;
  iconBg: string;
  iconColor: string;
  features: { text: string; included: boolean }[];
  cta: string;
  popular: boolean;
}

const plans: Plan[] = [
  {
    id: 'free',
    name: 'Free',
    description: 'Perfect for trying out SonZo AI',
    price: { monthly: 0, annual: 0 },
    icon: <Sparkles className="w-6 h-6" />,
    color: 'from-slate-500/10 to-slate-500/5',
    borderColor: 'border-slate-500/20',
    iconBg: 'bg-slate-500/20',
    iconColor: 'text-slate-500',
    features: [
      { text: '10 API calls per month', included: true },
      { text: 'Webcam support only', included: true },
      { text: 'ASL recognition', included: true },
      { text: 'Basic avatar responses', included: true },
      { text: 'OAK AI camera support', included: false },
      { text: 'Lumen 3D camera support', included: false },
      { text: 'All 12 sign languages', included: false },
      { text: 'Priority support', included: false },
    ],
    cta: 'Get Started Free',
    popular: false,
  },
  {
    id: 'pro',
    name: 'Pro',
    description: 'For professionals and educators',
    price: { monthly: 299, annual: 2990 },
    icon: <Zap className="w-6 h-6" />,
    color: 'from-violet-500/10 to-violet-500/5',
    borderColor: 'border-violet-500/30',
    iconBg: 'bg-violet-500/20',
    iconColor: 'text-violet-500',
    features: [
      { text: '100 API calls per month', included: true },
      { text: 'Webcam support', included: true },
      { text: 'OAK AI camera support', included: true },
      { text: 'Lumen 3D camera support', included: true },
      { text: 'All 12 sign languages', included: true },
      { text: 'Advanced avatar with depth', included: true },
      { text: 'Priority email support', included: true },
      { text: 'API access', included: true },
    ],
    cta: 'Start Pro Plan',
    popular: true,
  },
  {
    id: 'enterprise',
    name: 'Enterprise',
    description: 'For organizations and institutions',
    price: { monthly: 1999, annual: 19990 },
    icon: <Crown className="w-6 h-6" />,
    color: 'from-yellow-500/10 to-yellow-500/5',
    borderColor: 'border-yellow-500/30',
    iconBg: 'bg-yellow-500/20',
    iconColor: 'text-yellow-500',
    features: [
      { text: 'Unlimited API calls', included: true },
      { text: 'All camera types supported', included: true },
      { text: 'All 12 sign languages', included: true },
      { text: 'Custom avatar branding', included: true },
      { text: 'Dedicated account manager', included: true },
      { text: 'SLA guarantee (99.9%)', included: true },
      { text: 'Custom integrations', included: true },
      { text: 'On-premise deployment option', included: true },
    ],
    cta: 'Contact Sales',
    popular: false,
  },
];

const Pricing: React.FC = () => {
  const { isAuthenticated } = useAuth();
  const [billingCycle, setBillingCycle] = useState<'monthly' | 'annual'>('monthly');
  const [loadingPlan, setLoadingPlan] = useState<string | null>(null);
  const [error, setError] = useState('');

  const handleSelectPlan = async (plan: Plan) => {
    if (plan.id === 'free') {
      window.location.href = '/signin';
      return;
    }

    if (plan.id === 'enterprise') {
      window.location.href = 'mailto:sales@sonzo.ai?subject=Enterprise Plan Inquiry';
      return;
    }

    if (!isAuthenticated) {
      window.location.href = '/signin';
      return;
    }

    setLoadingPlan(plan.id);
    setError('');

    try {
      const token = localStorage.getItem('sonzo_token');
      const { data, error: fnError } = await supabase.functions.invoke('stripe-checkout', {
        body: {
          token,
          plan_id: plan.id,
          billing_cycle: billingCycle,
        },
      });

      if (fnError || data?.error) {
        setError(data?.error || 'Failed to create checkout session');
        return;
      }

      if (data?.checkout_url) {
        window.location.href = data.checkout_url;
      }
    } catch (err) {
      setError('Failed to start checkout. Please try again.');
    } finally {
      setLoadingPlan(null);
    }
  };

  const handleManageBilling = async () => {
    try {
      const token = localStorage.getItem('sonzo_token');
      const { data, error: fnError } = await supabase.functions.invoke('stripe-portal', {
        body: { token },
      });

      if (fnError || data?.error) {
        setError(data?.error || 'Failed to open billing portal');
        return;
      }

      if (data?.portal_url) {
        window.location.href = data.portal_url;
      }
    } catch (err) {
      setError('Failed to open billing portal');
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Background effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-72 h-72 bg-violet-500/5 rounded-full blur-[100px]" />
        <div className="absolute top-40 right-20 w-96 h-96 bg-purple-500/5 rounded-full blur-[120px]" />
      </div>

      {/* Navigation */}
      <nav className="relative z-20 flex items-center justify-between p-6 lg:px-12">
        <Link to="/" className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
            <svg viewBox="0 0 24 24" className="w-7 h-7 text-white" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M7 11V7a5 5 0 0 1 10 0v4" strokeLinecap="round" />
              <path d="M12 11v6" strokeLinecap="round" />
              <path d="M8 15h8" strokeLinecap="round" />
              <circle cx="12" cy="18" r="1" fill="currentColor" />
              <path d="M5 11h2v8a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2v-8h2" strokeLinecap="round" />
            </svg>
          </div>
          <div>
            <span className="font-bold text-2xl tracking-tight">
              Son<span className="text-violet-500">Zo</span>
            </span>
            <span className="block text-xs text-muted-foreground">AI Communication</span>
          </div>
        </Link>

        <div className="flex items-center gap-4">
          <Link to="/" className="text-sm text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1">
            <ArrowLeft className="w-4 h-4" />
            Back
          </Link>
          {isAuthenticated ? (
            <button
              onClick={handleManageBilling}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-muted hover:bg-muted/80 text-sm font-medium transition-colors"
            >
              <CreditCard className="w-4 h-4" />
              Manage Billing
            </button>
          ) : (
            <Link
              to="/signin"
              className="px-6 py-2 rounded-lg bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
            >
              Sign In
            </Link>
          )}
        </div>
      </nav>

      {/* Header */}
      <section className="relative z-10 px-6 lg:px-12 pt-12 pb-16 text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-violet-500/10 border border-violet-500/20 mb-6">
          <Sparkles className="w-4 h-4 text-violet-500" />
          <span className="text-sm text-violet-500 font-medium">Simple, transparent pricing</span>
        </div>

        <h1 className="font-extrabold text-4xl md:text-6xl tracking-tight mb-4">
          Choose Your
          <span className="block gradient-text">Plan</span>
        </h1>

        <p className="text-lg text-muted-foreground max-w-xl mx-auto mb-8">
          Start free and upgrade as you grow. All plans include core ASL recognition.
        </p>

        {/* Error message */}
        {error && (
          <div className="max-w-md mx-auto mb-6 p-3 rounded-lg bg-red-500/10 text-red-500 text-sm">
            {error}
          </div>
        )}

        {/* Billing toggle */}
        <div className="inline-flex items-center gap-3 p-1 bg-muted rounded-xl mb-12">
          <button
            onClick={() => setBillingCycle('monthly')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              billingCycle === 'monthly'
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            Monthly
          </button>
          <button
            onClick={() => setBillingCycle('annual')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              billingCycle === 'annual'
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            Annual
            <span className="ml-2 px-2 py-0.5 bg-green-500/20 text-green-500 text-xs rounded-full">
              Save 17%
            </span>
          </button>
        </div>
      </section>

      {/* Plans */}
      <section className="relative z-10 px-6 lg:px-12 pb-24">
        <div className="max-w-6xl mx-auto grid md:grid-cols-3 gap-6 lg:gap-8">
          {plans.map((plan) => (
            <div
              key={plan.id}
              className={`relative bg-gradient-to-br ${plan.color} rounded-2xl p-6 lg:p-8 border ${plan.borderColor} ${
                plan.popular ? 'ring-2 ring-violet-500 scale-105' : ''
              } transition-transform hover:scale-[1.02]`}
            >
              {plan.popular && (
                <div className="absolute -top-4 left-1/2 -translate-x-1/2 px-4 py-1 bg-violet-500 text-white text-sm font-medium rounded-full">
                  Most Popular
                </div>
              )}

              <div className="flex items-center gap-3 mb-4">
                <div className={`p-3 rounded-xl ${plan.iconBg}`}>
                  <div className={plan.iconColor}>{plan.icon}</div>
                </div>
                <div>
                  <h3 className="text-xl font-bold">{plan.name}</h3>
                  <p className="text-sm text-muted-foreground">{plan.description}</p>
                </div>
              </div>

              <div className="mb-6">
                <div className="flex items-baseline gap-1">
                  <span className="text-4xl font-bold">
                    ${billingCycle === 'monthly' ? plan.price.monthly : Math.round(plan.price.annual / 12)}
                  </span>
                  <span className="text-muted-foreground">/month</span>
                </div>
                {billingCycle === 'annual' && plan.price.annual > 0 && (
                  <p className="text-sm text-muted-foreground mt-1">
                    ${plan.price.annual.toLocaleString()} billed annually
                  </p>
                )}
              </div>

              <ul className="space-y-3 mb-8">
                {plan.features.map((feature, i) => (
                  <li key={i} className="flex items-start gap-3">
                    {feature.included ? (
                      <Check className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                    ) : (
                      <X className="w-5 h-5 text-muted-foreground flex-shrink-0 mt-0.5" />
                    )}
                    <span className={feature.included ? '' : 'text-muted-foreground'}>
                      {feature.text}
                    </span>
                  </li>
                ))}
              </ul>

              <button
                onClick={() => handleSelectPlan(plan)}
                disabled={loadingPlan === plan.id}
                className={`w-full flex items-center justify-center gap-2 py-3 px-6 rounded-xl font-medium transition-colors disabled:opacity-50 ${
                  plan.popular
                    ? 'bg-violet-500 text-white hover:bg-violet-600'
                    : plan.id === 'enterprise'
                    ? 'bg-yellow-500/20 text-yellow-500 hover:bg-yellow-500/30 border border-yellow-500/30'
                    : 'bg-muted hover:bg-muted/80'
                }`}
              >
                {loadingPlan === plan.id ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <>
                    {plan.cta}
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </div>
          ))}
        </div>
      </section>

      {/* Camera Support Info */}
      <section className="relative z-10 px-6 lg:px-12 pb-24">
        <div className="max-w-6xl mx-auto bg-muted/30 rounded-2xl p-8 border border-border">
          <h3 className="text-xl font-bold mb-6 text-center">Supported Camera Types</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="flex items-start gap-4">
              <div className="p-3 bg-blue-500/20 rounded-xl">
                <Monitor className="w-6 h-6 text-blue-500" />
              </div>
              <div>
                <h4 className="font-semibold">Webcam / Phone Camera</h4>
                <p className="text-sm text-muted-foreground mt-1">
                  Works with any standard webcam or smartphone camera.
                </p>
                <p className="text-xs text-green-500 mt-2">Available on all plans</p>
              </div>
            </div>
            <div className="flex items-start gap-4">
              <div className="p-3 bg-violet-500/20 rounded-xl">
                <Zap className="w-6 h-6 text-violet-500" />
              </div>
              <div>
                <h4 className="font-semibold">Luxonis OAK-D Pro</h4>
                <p className="text-sm text-muted-foreground mt-1">
                  Professional depth-sensing camera with stereo 3D vision.
                </p>
                <p className="text-xs text-violet-500 mt-2">Pro & Enterprise plans</p>
              </div>
            </div>
            <div className="flex items-start gap-4">
              <div className="p-3 bg-cyan-500/20 rounded-xl">
                <Video className="w-6 h-6 text-cyan-500" />
              </div>
              <div>
                <h4 className="font-semibold">Lumen 3D Camera</h4>
                <p className="text-sm text-muted-foreground mt-1">
                  High-precision 3D capture with advanced depth mapping.
                </p>
                <p className="text-xs text-cyan-500 mt-2">Pro & Enterprise plans</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Trust section */}
      <section className="relative z-10 px-6 lg:px-12 pb-24">
        <div className="max-w-4xl mx-auto text-center">
          <div className="flex flex-wrap justify-center gap-8 text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-green-500" />
              <span>Secure payments via Stripe</span>
            </div>
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-violet-500" />
              <span>Cancel anytime</span>
            </div>
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-yellow-500" />
              <span>Instant access after payment</span>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 px-6 lg:px-12 py-12 border-t border-border">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
              <svg viewBox="0 0 24 24" className="w-6 h-6 text-white" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M7 11V7a5 5 0 0 1 10 0v4" strokeLinecap="round" />
                <path d="M12 11v6" strokeLinecap="round" />
              </svg>
            </div>
            <div>
              <span className="font-bold text-lg">Son<span className="text-violet-500">Zo</span></span>
              <span className="block text-xs text-muted-foreground">AI Communication</span>
            </div>
          </div>
          <p className="text-sm text-muted-foreground">
            &copy; 2025 SonZo AI. Patent Pending. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Pricing;
