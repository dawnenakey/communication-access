import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  ArrowLeft, ArrowRight, Loader2, CheckCircle2, Building2,
  User, Mail, Phone, MessageSquare, Calendar, Sparkles, Hand
} from 'lucide-react';

interface FormData {
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  organization: string;
  role: string;
  serviceType: string;
  useCase: string;
  timeline: string;
  additionalNotes: string;
}

const initialFormData: FormData = {
  firstName: '',
  lastName: '',
  email: '',
  phone: '',
  organization: '',
  role: '',
  serviceType: '',
  useCase: '',
  timeline: '',
  additionalNotes: '',
};

const serviceTypes = [
  { value: 'enterprise', label: 'Enterprise Solution' },
  { value: 'education', label: 'Education & Training' },
  { value: 'healthcare', label: 'Healthcare' },
  { value: 'government', label: 'Government & Public Sector' },
  { value: 'events', label: 'Events & Conferences' },
  { value: 'broadcasting', label: 'Broadcasting & Media' },
  { value: 'custom', label: 'Custom Integration' },
  { value: 'other', label: 'Other' },
];

const timelines = [
  { value: 'immediate', label: 'Immediate (within 2 weeks)' },
  { value: '1-3months', label: '1-3 months' },
  { value: '3-6months', label: '3-6 months' },
  { value: '6months+', label: '6+ months' },
  { value: 'exploring', label: 'Just exploring options' },
];

const Intake: React.FC = () => {
  const [formData, setFormData] = useState<FormData>(initialFormData);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsSubmitting(true);

    // Basic validation
    if (!formData.firstName || !formData.lastName || !formData.email) {
      setError('Please fill in all required fields.');
      setIsSubmitting(false);
      return;
    }

    if (!formData.email.includes('@')) {
      setError('Please enter a valid email address.');
      setIsSubmitting(false);
      return;
    }

    try {
      // TODO: Replace with actual API call to submit intake form
      // For now, simulate submission
      await new Promise((resolve) => setTimeout(resolve, 1500));

      // Could send to backend, email service, or CRM here
      console.log('Intake form submitted:', formData);

      setIsSubmitted(true);
    } catch (err) {
      setError('Failed to submit form. Please try again or contact us directly.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const inputClasses =
    'w-full px-4 py-3 rounded-xl bg-background border border-border focus:border-violet-500 focus:ring-2 focus:ring-violet-500/20 outline-none transition-all';
  const labelClasses = 'block text-sm font-medium mb-2';

  if (isSubmitted) {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <div className="max-w-md mx-auto text-center px-6">
          <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-green-500/20 flex items-center justify-center">
            <CheckCircle2 className="w-10 h-10 text-green-500" />
          </div>
          <h1 className="text-3xl font-bold mb-4">Thank You!</h1>
          <p className="text-muted-foreground mb-8">
            We've received your inquiry and will get back to you within 1-2 business days.
            Our team is excited to learn more about how SonZo AI can help your organization.
          </p>
          <Link
            to="/"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-violet-500 text-white font-medium hover:bg-violet-600 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Home
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Background effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-72 h-72 bg-violet-500/5 rounded-full blur-[100px]" />
        <div className="absolute top-40 right-20 w-96 h-96 bg-purple-500/5 rounded-full blur-[120px]" />
        <div className="absolute bottom-20 left-1/3 w-80 h-80 bg-cyan-500/5 rounded-full blur-[100px]" />
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
          <Link
            to="/pricing"
            className="px-6 py-2 rounded-lg bg-muted hover:bg-muted/80 font-medium text-sm transition-colors"
          >
            View Pricing
          </Link>
        </div>
      </nav>

      {/* Header */}
      <section className="relative z-10 px-6 lg:px-12 pt-8 pb-12 text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-violet-500/10 border border-violet-500/20 mb-6">
          <Hand className="w-4 h-4 text-violet-500" />
          <span className="text-sm text-violet-500 font-medium">Let's connect</span>
        </div>

        <h1 className="font-extrabold text-4xl md:text-5xl tracking-tight mb-4">
          Get Started with
          <span className="block gradient-text">SonZo AI</span>
        </h1>

        <p className="text-lg text-muted-foreground max-w-xl mx-auto">
          Tell us about your needs and we'll help you find the perfect solution
          for accessible communication.
        </p>
      </section>

      {/* Form */}
      <section className="relative z-10 px-6 lg:px-12 pb-24">
        <div className="max-w-2xl mx-auto">
          <form onSubmit={handleSubmit} className="bg-muted/30 rounded-2xl p-8 border border-border">
            {/* Error message */}
            {error && (
              <div className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-500 text-sm">
                {error}
              </div>
            )}

            {/* Contact Information */}
            <div className="mb-8">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <User className="w-5 h-5 text-violet-500" />
                Contact Information
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="firstName" className={labelClasses}>
                    First Name <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="text"
                    id="firstName"
                    name="firstName"
                    value={formData.firstName}
                    onChange={handleChange}
                    placeholder="John"
                    className={inputClasses}
                    required
                  />
                </div>
                <div>
                  <label htmlFor="lastName" className={labelClasses}>
                    Last Name <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="text"
                    id="lastName"
                    name="lastName"
                    value={formData.lastName}
                    onChange={handleChange}
                    placeholder="Doe"
                    className={inputClasses}
                    required
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <div>
                  <label htmlFor="email" className={labelClasses}>
                    Email Address <span className="text-red-500">*</span>
                  </label>
                  <div className="relative">
                    <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                    <input
                      type="email"
                      id="email"
                      name="email"
                      value={formData.email}
                      onChange={handleChange}
                      placeholder="john@company.com"
                      className={`${inputClasses} pl-12`}
                      required
                    />
                  </div>
                </div>
                <div>
                  <label htmlFor="phone" className={labelClasses}>
                    Phone Number
                  </label>
                  <div className="relative">
                    <Phone className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                    <input
                      type="tel"
                      id="phone"
                      name="phone"
                      value={formData.phone}
                      onChange={handleChange}
                      placeholder="+1 (555) 123-4567"
                      className={`${inputClasses} pl-12`}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Organization Information */}
            <div className="mb-8">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Building2 className="w-5 h-5 text-violet-500" />
                Organization
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="organization" className={labelClasses}>
                    Organization Name
                  </label>
                  <input
                    type="text"
                    id="organization"
                    name="organization"
                    value={formData.organization}
                    onChange={handleChange}
                    placeholder="Acme Corporation"
                    className={inputClasses}
                  />
                </div>
                <div>
                  <label htmlFor="role" className={labelClasses}>
                    Your Role
                  </label>
                  <input
                    type="text"
                    id="role"
                    name="role"
                    value={formData.role}
                    onChange={handleChange}
                    placeholder="Accessibility Coordinator"
                    className={inputClasses}
                  />
                </div>
              </div>
            </div>

            {/* Service Details */}
            <div className="mb-8">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-violet-500" />
                Service Details
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="serviceType" className={labelClasses}>
                    Service Type
                  </label>
                  <select
                    id="serviceType"
                    name="serviceType"
                    value={formData.serviceType}
                    onChange={handleChange}
                    className={inputClasses}
                  >
                    <option value="">Select a service...</option>
                    {serviceTypes.map((type) => (
                      <option key={type.value} value={type.value}>
                        {type.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label htmlFor="timeline" className={labelClasses}>
                    <Calendar className="w-4 h-4 inline mr-1" />
                    Expected Timeline
                  </label>
                  <select
                    id="timeline"
                    name="timeline"
                    value={formData.timeline}
                    onChange={handleChange}
                    className={inputClasses}
                  >
                    <option value="">Select timeline...</option>
                    {timelines.map((t) => (
                      <option key={t.value} value={t.value}>
                        {t.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="mt-4">
                <label htmlFor="useCase" className={labelClasses}>
                  <MessageSquare className="w-4 h-4 inline mr-1" />
                  Describe Your Use Case
                </label>
                <textarea
                  id="useCase"
                  name="useCase"
                  value={formData.useCase}
                  onChange={handleChange}
                  placeholder="Tell us about your organization's communication needs and how you envision using SonZo AI..."
                  rows={4}
                  className={inputClasses}
                />
              </div>

              <div className="mt-4">
                <label htmlFor="additionalNotes" className={labelClasses}>
                  Additional Notes
                </label>
                <textarea
                  id="additionalNotes"
                  name="additionalNotes"
                  value={formData.additionalNotes}
                  onChange={handleChange}
                  placeholder="Any other information you'd like to share..."
                  rows={3}
                  className={inputClasses}
                />
              </div>
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full flex items-center justify-center gap-2 py-4 px-6 rounded-xl bg-gradient-to-r from-violet-500 to-purple-600 text-white font-medium hover:from-violet-600 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Submitting...
                </>
              ) : (
                <>
                  Submit Inquiry
                  <ArrowRight className="w-5 h-5" />
                </>
              )}
            </button>

            <p className="text-center text-sm text-muted-foreground mt-4">
              We respect your privacy. Your information will only be used to respond to your inquiry.
            </p>
          </form>
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

export default Intake;
