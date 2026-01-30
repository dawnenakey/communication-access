import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import {
  Mail, Lock, User, Eye, EyeOff, ArrowRight,
  Loader2, AlertCircle, CheckCircle, Hand, Sparkles
} from 'lucide-react';

const SignIn: React.FC = () => {
  const { signIn, signUp } = useAuth();
  const [mode, setMode] = useState<'signin' | 'signup'>('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const resetForm = () => {
    setEmail('');
    setPassword('');
    setConfirmPassword('');
    setDisplayName('');
    setError('');
    setSuccess('');
  };

  const handleModeSwitch = (newMode: 'signin' | 'signup') => {
    setMode(newMode);
    resetForm();
  };

  const validateForm = () => {
    if (!email || !password) {
      setError('Please fill in all required fields');
      return false;
    }
    if (!email.includes('@')) {
      setError('Please enter a valid email address');
      return false;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters');
      return false;
    }
    if (mode === 'signup' && password !== confirmPassword) {
      setError('Passwords do not match');
      return false;
    }
    return true;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    if (!validateForm()) return;

    setIsLoading(true);

    try {
      if (mode === 'signup') {
        const result = await signUp(email, password, displayName || undefined);
        if (result.success) {
          setSuccess('Account created successfully! Redirecting...');
          setTimeout(() => {
            window.location.href = '/';
          }, 1500);
        } else {
          setError(result.error || 'Sign up failed');
        }
      } else {
        const result = await signIn(email, password);
        if (result.success) {
          setSuccess('Signed in successfully! Redirecting...');
          setTimeout(() => {
            window.location.href = '/';
          }, 1000);
        } else {
          setError(result.error || 'Sign in failed');
        }
      }
    } catch (err) {
      setError('An unexpected error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex">
      {/* Left side - branding */}
      <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden bg-gradient-to-br from-violet-500/10 via-purple-500/5 to-background">
        <div className="absolute top-20 left-10 w-72 h-72 bg-violet-500/10 rounded-full blur-[100px]" />
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-purple-500/10 rounded-full blur-[120px]" />

        <div className="relative z-10 flex flex-col justify-center px-16">
          <Link to="/" className="flex items-center gap-3 mb-12">
            <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
              <svg viewBox="0 0 24 24" className="w-8 h-8 text-white" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M7 11V7a5 5 0 0 1 10 0v4" strokeLinecap="round" />
                <path d="M12 11v6" strokeLinecap="round" />
                <path d="M8 15h8" strokeLinecap="round" />
                <circle cx="12" cy="18" r="1" fill="currentColor" />
                <path d="M5 11h2v8a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2v-8h2" strokeLinecap="round" />
              </svg>
            </div>
            <div>
              <span className="font-bold text-3xl tracking-tight">
                Son<span className="text-violet-500">Zo</span>
              </span>
              <span className="block text-sm text-muted-foreground">AI Communication</span>
            </div>
          </Link>

          <h1 className="font-extrabold text-4xl tracking-tight leading-tight mb-6">
            Breaking Barriers in
            <br />
            <span className="gradient-text">Sign Language</span>
            <br />
            Communication
          </h1>

          <p className="text-lg text-muted-foreground leading-relaxed max-w-md mb-8">
            Real-time ASL recognition, translation, and bidirectional communication
            powered by AI.
          </p>

          <div className="flex gap-6 text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <Hand className="w-4 h-4 text-violet-500" />
              <span>12 Sign Languages</span>
            </div>
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-purple-500" />
              <span>AI Powered</span>
            </div>
          </div>
        </div>
      </div>

      {/* Right side - form */}
      <div className="flex-1 flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-md">
          {/* Mobile logo */}
          <Link to="/" className="flex items-center gap-3 mb-10 lg:hidden">
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

          <div className="mb-8">
            <h2 className="font-bold text-3xl tracking-tight mb-2">
              {mode === 'signup' ? 'Create an account' : 'Welcome back'}
            </h2>
            <p className="text-muted-foreground">
              {mode === 'signup'
                ? 'Sign up to start using SonZo AI'
                : 'Sign in to your SonZo AI account'}
            </p>
          </div>

          {/* Tabs */}
          <div className="flex border-b border-border mb-6">
            <button
              onClick={() => handleModeSwitch('signin')}
              className={`flex-1 py-3 text-sm font-medium transition-colors ${
                mode === 'signin'
                  ? 'text-primary border-b-2 border-primary'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              Sign In
            </button>
            <button
              onClick={() => handleModeSwitch('signup')}
              className={`flex-1 py-3 text-sm font-medium transition-colors ${
                mode === 'signup'
                  ? 'text-primary border-b-2 border-primary'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              Sign Up
            </button>
          </div>

          {/* Error/Success messages */}
          {error && (
            <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 text-red-500 text-sm mb-4">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {error}
            </div>
          )}
          {success && (
            <div className="flex items-center gap-2 p-3 rounded-lg bg-green-500/10 text-green-500 text-sm mb-4">
              <CheckCircle className="w-4 h-4 flex-shrink-0" />
              {success}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            {mode === 'signup' && (
              <div>
                <label className="block text-sm font-medium mb-1.5">Display Name</label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                  <input
                    type="text"
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    placeholder="Your name"
                    className="w-full pl-11 pr-4 py-3 rounded-xl bg-background border border-border focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none transition-all"
                  />
                </div>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium mb-1.5">Email</label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  required
                  className="w-full pl-11 pr-4 py-3 rounded-xl bg-background border border-border focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none transition-all"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1.5">Password</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder={mode === 'signup' ? 'Min 6 characters' : 'Your password'}
                  required
                  className="w-full pl-11 pr-12 py-3 rounded-xl bg-background border border-border focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none transition-all"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-muted-foreground hover:text-foreground"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>

            {mode === 'signup' && (
              <div>
                <label className="block text-sm font-medium mb-1.5">Confirm Password</label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="Confirm your password"
                    required
                    className="w-full pl-11 pr-4 py-3 rounded-xl bg-background border border-border focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none transition-all"
                  />
                </div>
              </div>
            )}

            <button
              type="submit"
              disabled={isLoading}
              className="w-full flex items-center justify-center gap-2 py-3 bg-primary text-primary-foreground rounded-xl font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  {mode === 'signin' ? 'Signing in...' : 'Creating account...'}
                </>
              ) : (
                <>
                  {mode === 'signin' ? 'Sign In' : 'Create Account'}
                  <ArrowRight className="w-4 h-4 ml-1" />
                </>
              )}
            </button>
          </form>

          <div className="mt-8 text-center">
            <Link
              to="/pricing"
              className="text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              View pricing plans
            </Link>
          </div>

          {/* Benefits */}
          <div className="mt-6 p-4 rounded-xl bg-muted/50">
            <p className="text-sm font-medium mb-2">
              {mode === 'signin' ? 'Your progress awaits!' : 'Create an account to:'}
            </p>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                Save your learning progress
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                Sync settings across devices
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                Access premium features with Pro plan
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SignIn;
