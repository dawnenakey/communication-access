import React, { useState } from 'react';
import { Check, X, Zap, Crown, Sparkles, Camera, Monitor, Video } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

interface PricingSectionProps {
  onSelectPlan?: (plan: string) => void;
}

const PricingSection: React.FC<PricingSectionProps> = ({ onSelectPlan }) => {
  const { isAuthenticated } = useAuth();
  const [billingCycle, setBillingCycle] = useState<'monthly' | 'annual'>('monthly');

  const plans = [
    {
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
      popular: false
    },
    {
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
      cta: 'Start Pro Trial',
      popular: true
    },
    {
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
      popular: false
    }
  ];

  const handleSelectPlan = (planName: string) => {
    if (onSelectPlan) {
      onSelectPlan(planName.toLowerCase());
    }
  };

  return (
    <section id="pricing" className="py-20 bg-gradient-to-b from-background to-muted/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            Simple, Transparent <span className="gradient-text">Pricing</span>
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-8">
            Choose the plan that fits your needs. All plans include our core sign language recognition technology.
          </p>

          {/* Billing Toggle */}
          <div className="inline-flex items-center gap-3 p-1 bg-muted rounded-xl">
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
        </div>

        {/* Pricing Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8">
          {plans.map((plan) => (
            <div
              key={plan.name}
              className={`relative bg-gradient-to-br ${plan.color} rounded-2xl p-6 lg:p-8 border ${plan.borderColor} ${
                plan.popular ? 'ring-2 ring-violet-500 scale-105' : ''
              } transition-transform hover:scale-[1.02]`}
            >
              {plan.popular && (
                <div className="absolute -top-4 left-1/2 -translate-x-1/2 px-4 py-1 bg-violet-500 text-white text-sm font-medium rounded-full">
                  Most Popular
                </div>
              )}

              {/* Plan Header */}
              <div className="flex items-center gap-3 mb-4">
                <div className={`p-3 rounded-xl ${plan.iconBg}`}>
                  <div className={plan.iconColor}>{plan.icon}</div>
                </div>
                <div>
                  <h3 className="text-xl font-bold">{plan.name}</h3>
                  <p className="text-sm text-muted-foreground">{plan.description}</p>
                </div>
              </div>

              {/* Price */}
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

              {/* Features */}
              <ul className="space-y-3 mb-8">
                {plan.features.map((feature, index) => (
                  <li key={index} className="flex items-start gap-3">
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

              {/* CTA Button */}
              <button
                onClick={() => handleSelectPlan(plan.name)}
                className={`w-full py-3 px-6 rounded-xl font-medium transition-colors ${
                  plan.popular
                    ? 'bg-violet-500 text-white hover:bg-violet-600'
                    : plan.name === 'Enterprise'
                    ? 'bg-yellow-500/20 text-yellow-500 hover:bg-yellow-500/30 border border-yellow-500/30'
                    : 'bg-muted hover:bg-muted/80'
                }`}
              >
                {plan.cta}
              </button>
            </div>
          ))}
        </div>

        {/* Camera Support Info */}
        <div className="mt-16 bg-muted/30 rounded-2xl p-8 border border-border">
          <h3 className="text-xl font-bold mb-6 text-center">Supported Camera Types</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="flex items-start gap-4">
              <div className="p-3 bg-blue-500/20 rounded-xl">
                <Monitor className="w-6 h-6 text-blue-500" />
              </div>
              <div>
                <h4 className="font-semibold">Webcam / Phone Camera</h4>
                <p className="text-sm text-muted-foreground mt-1">
                  Works with any standard webcam or smartphone camera. 2D RGB input with AI-powered hand tracking.
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
                  Professional depth-sensing camera with stereo 3D vision. Enhanced accuracy for complex signs.
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
                  High-precision 3D capture with advanced depth mapping. Ideal for research and enterprise use.
                </p>
                <p className="text-xs text-cyan-500 mt-2">Pro & Enterprise plans</p>
              </div>
            </div>
          </div>
        </div>

        {/* FAQ Link */}
        <div className="text-center mt-12">
          <p className="text-muted-foreground">
            Have questions? Check our{' '}
            <a href="#faq" className="text-primary hover:underline">
              FAQ section
            </a>{' '}
            or{' '}
            <a href="mailto:support@sonzo.ai" className="text-primary hover:underline">
              contact support
            </a>
          </p>
        </div>
      </div>
    </section>
  );
};

export default PricingSection;
