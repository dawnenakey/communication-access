import React, { useState } from 'react';
import { 
  Mail, MapPin, Phone, Send, 
  Github, Twitter, Linkedin, Youtube,
  ExternalLink, Heart, Globe, Shield
} from 'lucide-react';

const Footer: React.FC = () => {
  const [email, setEmail] = useState('');
  const [subscribed, setSubscribed] = useState(false);

  const handleSubscribe = (e: React.FormEvent) => {
    e.preventDefault();
    if (email) {
      setSubscribed(true);
      setEmail('');
      setTimeout(() => setSubscribed(false), 3000);
    }
  };

  const footerLinks = {
    product: [
      { label: 'Features', href: '#features' },
      { label: 'Recognition', href: '#recognition' },
      { label: 'Learn', href: '#learn' },
      { label: 'Dashboard', href: '#dashboard' },
      { label: 'Pricing', href: '#pricing' },
      { label: 'API Access', href: '#api' }
    ],
    languages: [
      { label: 'ASL (American)', href: '#' },
      { label: 'BSL (British)', href: '#' },
      { label: 'ISL (Indian)', href: '#' },
      { label: 'LSF (French)', href: '#' },
      { label: 'DGS (German)', href: '#' },
      { label: 'View All 12+', href: '#' }
    ],
    resources: [
      { label: 'Documentation', href: '#' },
      { label: 'API Reference', href: '#' },
      { label: 'Tutorials', href: '#' },
      { label: 'Blog', href: '#' },
      { label: 'Community', href: '#' },
      { label: 'Support', href: '#' }
    ],
    company: [
      { label: 'About Us', href: '#' },
      { label: 'Careers', href: '#' },
      { label: 'Press Kit', href: '#' },
      { label: 'Partners', href: '#' },
      { label: 'Contact', href: '#contact' },
      { label: 'Legal', href: '#' }
    ]
  };

  const socialLinks = [
    { icon: Twitter, href: '#', label: 'Twitter' },
    { icon: Github, href: '#', label: 'GitHub' },
    { icon: Linkedin, href: '#', label: 'LinkedIn' },
    { icon: Youtube, href: '#', label: 'YouTube' }
  ];

  return (
    <footer className="bg-card border-t border-border">
      {/* Newsletter Section */}
      <div className="border-b border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid md:grid-cols-2 gap-8 items-center">
            <div>
              <h3 className="text-2xl font-bold mb-2">Stay Updated</h3>
              <p className="text-muted-foreground">
                Get the latest news on new sign languages, features, and AI improvements.
              </p>
            </div>
            <form onSubmit={handleSubscribe} className="flex gap-3">
              <div className="flex-1 relative">
                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Enter your email"
                  className="w-full pl-12 pr-4 py-3 rounded-xl bg-background border border-border focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none transition-all"
                  required
                />
              </div>
              <button
                type="submit"
                className="px-6 py-3 bg-primary text-primary-foreground rounded-xl font-medium hover:bg-primary/90 transition-colors flex items-center gap-2"
              >
                <Send className="w-4 h-4" />
                <span className="hidden sm:inline">Subscribe</span>
              </button>
            </form>
            {subscribed && (
              <p className="text-green-500 text-sm md:col-span-2">
                Thanks for subscribing! Check your email for confirmation.
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Main Footer */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-2 md:grid-cols-6 gap-8">
          {/* Brand Column */}
          <div className="col-span-2">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                <svg viewBox="0 0 24 24" className="w-6 h-6 text-white" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M7 11V7a5 5 0 0 1 10 0v4" strokeLinecap="round" />
                  <path d="M12 11v6" strokeLinecap="round" />
                  <path d="M8 15h8" strokeLinecap="round" />
                  <circle cx="12" cy="18" r="1" fill="currentColor" />
                  <path d="M5 11h2v8a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2v-8h2" strokeLinecap="round" />
                </svg>
              </div>
              <div>
                <h4 className="font-bold text-lg">SonZo AI</h4>
                <p className="text-xs text-muted-foreground">Sign Language Recognition</p>
              </div>
            </div>
            <p className="text-sm text-muted-foreground mb-4 max-w-xs">
              Breaking communication barriers with AI-powered sign language recognition. 
              Supporting 12+ languages globally.
            </p>

            {/* Contact Info */}
            <div className="space-y-2 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <Mail className="w-4 h-4" />
                <a href="mailto:hello@sonzo.ai" className="hover:text-primary transition-colors">
                  hello@sonzo.ai
                </a>
              </div>
              <div className="flex items-center gap-2">
                <MapPin className="w-4 h-4" />
                <span>San Francisco, CA</span>
              </div>
            </div>

            {/* Social Links */}
            <div className="flex gap-3 mt-4">
              {socialLinks.map((social) => (
                <a
                  key={social.label}
                  href={social.href}
                  aria-label={social.label}
                  className="w-10 h-10 rounded-lg bg-muted hover:bg-primary hover:text-primary-foreground flex items-center justify-center transition-colors"
                >
                  <social.icon className="w-5 h-5" />
                </a>
              ))}
            </div>
          </div>

          {/* Product Links */}
          <div>
            <h5 className="font-semibold mb-4">Product</h5>
            <ul className="space-y-2">
              {footerLinks.product.map((link) => (
                <li key={link.label}>
                  <a
                    href={link.href}
                    className="text-sm text-muted-foreground hover:text-primary transition-colors"
                  >
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Languages Links */}
          <div>
            <h5 className="font-semibold mb-4">Languages</h5>
            <ul className="space-y-2">
              {footerLinks.languages.map((link) => (
                <li key={link.label}>
                  <a
                    href={link.href}
                    className="text-sm text-muted-foreground hover:text-primary transition-colors"
                  >
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Resources Links */}
          <div>
            <h5 className="font-semibold mb-4">Resources</h5>
            <ul className="space-y-2">
              {footerLinks.resources.map((link) => (
                <li key={link.label}>
                  <a
                    href={link.href}
                    className="text-sm text-muted-foreground hover:text-primary transition-colors flex items-center gap-1"
                  >
                    {link.label}
                    {link.label === 'Documentation' && <ExternalLink className="w-3 h-3" />}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Company Links */}
          <div>
            <h5 className="font-semibold mb-4">Company</h5>
            <ul className="space-y-2">
              {footerLinks.company.map((link) => (
                <li key={link.label}>
                  <a
                    href={link.href}
                    className="text-sm text-muted-foreground hover:text-primary transition-colors"
                  >
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Bottom Bar */}
      <div className="border-t border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex flex-wrap items-center justify-center gap-4 text-sm text-muted-foreground">
              <span>&copy; {new Date().getFullYear()} SonZo AI. All rights reserved.</span>
              <span className="hidden md:inline">|</span>
              <a href="#" className="hover:text-primary transition-colors">Privacy Policy</a>
              <a href="#" className="hover:text-primary transition-colors">Terms of Service</a>
              <a href="#" className="hover:text-primary transition-colors">Cookie Policy</a>
            </div>

            <div className="flex items-center gap-4">
              {/* Trust Badges */}
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-muted text-xs">
                <Shield className="w-4 h-4 text-green-500" />
                <span>GDPR Compliant</span>
              </div>
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-muted text-xs">
                <Globe className="w-4 h-4 text-blue-500" />
                <span>12+ Languages</span>
              </div>
            </div>
          </div>

          {/* Made with love */}
          <div className="text-center mt-6 text-sm text-muted-foreground">
            Made with <Heart className="w-4 h-4 inline text-red-500 mx-1" /> for the deaf and hard of hearing community
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
