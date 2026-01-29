import React, { useState } from 'react';
import { 
  Globe, Plus, Check, Users, 
  ArrowRight, MessageSquare, BookOpen,
  Mail, ExternalLink
} from 'lucide-react';

interface Language {
  code: string;
  name: string;
  nativeName: string;
  flag: string;
  users: string;
  sentences: number;
  status: 'available' | 'beta' | 'coming';
}

const languages: Language[] = [
  { code: 'ASL', name: 'American Sign Language', nativeName: 'ASL', flag: '🇺🇸', users: '500K+', sentences: 5000, status: 'available' },
  { code: 'BSL', name: 'British Sign Language', nativeName: 'BSL', flag: '🇬🇧', users: '150K+', sentences: 4200, status: 'available' },
  { code: 'ISL', name: 'Indian Sign Language', nativeName: 'भारतीय सांकेतिक भाषा', flag: '🇮🇳', users: '1.8M+', sentences: 3800, status: 'available' },
  { code: 'LSF', name: 'French Sign Language', nativeName: 'Langue des Signes Française', flag: '🇫🇷', users: '100K+', sentences: 3500, status: 'available' },
  { code: 'DGS', name: 'German Sign Language', nativeName: 'Deutsche Gebärdensprache', flag: '🇩🇪', users: '200K+', sentences: 3200, status: 'available' },
  { code: 'JSL', name: 'Japanese Sign Language', nativeName: '日本手話', flag: '🇯🇵', users: '320K+', sentences: 2800, status: 'available' },
  { code: 'Auslan', name: 'Australian Sign Language', nativeName: 'Auslan', flag: '🇦🇺', users: '30K+', sentences: 2500, status: 'available' },
  { code: 'LSM', name: 'Mexican Sign Language', nativeName: 'Lengua de Señas Mexicana', flag: '🇲🇽', users: '400K+', sentences: 2200, status: 'available' },
  { code: 'KSL', name: 'Korean Sign Language', nativeName: '한국 수어', flag: '🇰🇷', users: '250K+', sentences: 2000, status: 'beta' },
  { code: 'CSL', name: 'Chinese Sign Language', nativeName: '中国手语', flag: '🇨🇳', users: '20M+', sentences: 1800, status: 'beta' },
  { code: 'RSL', name: 'Russian Sign Language', nativeName: 'Русский жестовый язык', flag: '🇷🇺', users: '120K+', sentences: 1500, status: 'beta' },
  { code: 'LSB', name: 'Brazilian Sign Language', nativeName: 'Libras', flag: '🇧🇷', users: '5M+', sentences: 1200, status: 'beta' },
  { code: 'SASL', name: 'South African Sign Language', nativeName: 'SASL', flag: '🇿🇦', users: '600K+', sentences: 0, status: 'coming' },
  { code: 'NSL', name: 'Nigerian Sign Language', nativeName: 'NSL', flag: '🇳🇬', users: '300K+', sentences: 0, status: 'coming' },
  { code: 'IPSL', name: 'Indo-Pakistani Sign Language', nativeName: 'IPSL', flag: '🇵🇰', users: '2M+', sentences: 0, status: 'coming' },
];

const GlobalLanguagesSection: React.FC = () => {
  const [showContributeForm, setShowContributeForm] = useState(false);
  const [contributeEmail, setContributeEmail] = useState('');
  const [contributeLanguage, setContributeLanguage] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const availableCount = languages.filter(l => l.status === 'available').length;
  const betaCount = languages.filter(l => l.status === 'beta').length;
  const comingCount = languages.filter(l => l.status === 'coming').length;

  const handleContributeSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
    setTimeout(() => {
      setShowContributeForm(false);
      setSubmitted(false);
      setContributeEmail('');
      setContributeLanguage('');
    }, 2000);
  };

  const getStatusBadge = (status: Language['status']) => {
    switch (status) {
      case 'available':
        return <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-green-500/10 text-green-500">Available</span>;
      case 'beta':
        return <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-yellow-500/10 text-yellow-500">Beta</span>;
      case 'coming':
        return <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-blue-500/10 text-blue-500">Coming Soon</span>;
    }
  };

  return (
    <section className="py-20 bg-muted/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 mb-4">
            <Globe className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium text-primary">Global Coverage</span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            Sign Languages <span className="gradient-text">Worldwide</span>
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            SonZo AI supports sign languages from around the globe. Help us expand 
            coverage by contributing to languages in your community.
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-4 mb-12 max-w-2xl mx-auto">
          <div className="text-center p-4 bg-card rounded-xl border border-border">
            <p className="text-3xl font-bold text-green-500">{availableCount}</p>
            <p className="text-sm text-muted-foreground">Available</p>
          </div>
          <div className="text-center p-4 bg-card rounded-xl border border-border">
            <p className="text-3xl font-bold text-yellow-500">{betaCount}</p>
            <p className="text-sm text-muted-foreground">In Beta</p>
          </div>
          <div className="text-center p-4 bg-card rounded-xl border border-border">
            <p className="text-3xl font-bold text-blue-500">{comingCount}</p>
            <p className="text-sm text-muted-foreground">Coming Soon</p>
          </div>
        </div>

        {/* Languages Grid */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-12">
          {languages.map((lang) => (
            <div
              key={lang.code}
              className={`p-4 bg-card rounded-xl border border-border card-hover ${
                lang.status === 'coming' ? 'opacity-70' : ''
              }`}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <span className="text-2xl">{lang.flag}</span>
                  <div>
                    <h3 className="font-semibold">{lang.code}</h3>
                    <p className="text-xs text-muted-foreground">{lang.name}</p>
                  </div>
                </div>
                {getStatusBadge(lang.status)}
              </div>

              <p className="text-sm text-muted-foreground mb-3 italic">
                {lang.nativeName}
              </p>

              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-1 text-muted-foreground">
                  <Users className="w-4 h-4" />
                  <span>{lang.users} users</span>
                </div>
                {lang.sentences > 0 && (
                  <div className="flex items-center gap-1 text-muted-foreground">
                    <MessageSquare className="w-4 h-4" />
                    <span>{lang.sentences.toLocaleString()} sentences</span>
                  </div>
                )}
              </div>

              {lang.status === 'available' && (
                <button className="w-full mt-4 flex items-center justify-center gap-2 px-4 py-2 bg-primary/10 text-primary rounded-lg text-sm font-medium hover:bg-primary/20 transition-colors">
                  <BookOpen className="w-4 h-4" />
                  Start Learning
                </button>
              )}
            </div>
          ))}
        </div>

        {/* Contribute CTA */}
        <div className="bg-gradient-to-r from-violet-500/10 via-purple-500/10 to-fuchsia-500/10 rounded-2xl border border-primary/20 p-8">
          <div className="grid md:grid-cols-2 gap-8 items-center">
            <div>
              <h3 className="text-2xl font-bold mb-4">
                Help Us Add More Languages
              </h3>
              <p className="text-muted-foreground mb-6">
                Are you part of a deaf community or a sign language expert? 
                Join our global contributor program to help add and improve 
                support for sign languages in your region.
              </p>
              <ul className="space-y-2 mb-6">
                {[
                  'Contribute sentence datasets',
                  'Validate recognition accuracy',
                  'Help with regional variations',
                  'Earn contributor recognition'
                ].map((item, i) => (
                  <li key={i} className="flex items-center gap-2 text-sm">
                    <Check className="w-4 h-4 text-green-500" />
                    {item}
                  </li>
                ))}
              </ul>
              <button
                onClick={() => setShowContributeForm(true)}
                className="flex items-center gap-2 px-6 py-3 bg-primary text-primary-foreground rounded-xl font-medium hover:bg-primary/90 transition-colors"
              >
                <Plus className="w-5 h-5" />
                Become a Contributor
              </button>
            </div>

            {/* Contribute Form */}
            {showContributeForm ? (
              <div className="bg-card rounded-xl border border-border p-6">
                {submitted ? (
                  <div className="text-center py-8">
                    <div className="w-16 h-16 rounded-full bg-green-500/10 flex items-center justify-center mx-auto mb-4">
                      <Check className="w-8 h-8 text-green-500" />
                    </div>
                    <h4 className="font-semibold mb-2">Thank You!</h4>
                    <p className="text-sm text-muted-foreground">
                      We'll be in touch soon about contributing to {contributeLanguage || 'sign language'} support.
                    </p>
                  </div>
                ) : (
                  <form onSubmit={handleContributeSubmit}>
                    <h4 className="font-semibold mb-4">Join Our Contributor Program</h4>
                    
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium mb-1">Email</label>
                        <div className="relative">
                          <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                          <input
                            type="email"
                            value={contributeEmail}
                            onChange={(e) => setContributeEmail(e.target.value)}
                            placeholder="your@email.com"
                            required
                            className="w-full pl-10 pr-4 py-2 rounded-lg bg-background border border-border focus:border-primary outline-none"
                          />
                        </div>
                      </div>

                      <div>
                        <label className="block text-sm font-medium mb-1">Sign Language</label>
                        <input
                          type="text"
                          value={contributeLanguage}
                          onChange={(e) => setContributeLanguage(e.target.value)}
                          placeholder="e.g., Nigerian Sign Language"
                          required
                          className="w-full px-4 py-2 rounded-lg bg-background border border-border focus:border-primary outline-none"
                        />
                      </div>

                      <button
                        type="submit"
                        className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors"
                      >
                        Submit Interest
                        <ArrowRight className="w-4 h-4" />
                      </button>
                    </div>
                  </form>
                )}
              </div>
            ) : (
              <div className="bg-card rounded-xl border border-border p-6">
                <h4 className="font-semibold mb-4">Current Contributors</h4>
                <div className="space-y-3">
                  {[
                    { name: 'National Association of the Deaf', country: 'USA', flag: '🇺🇸' },
                    { name: 'British Deaf Association', country: 'UK', flag: '🇬🇧' },
                    { name: 'All India Federation of the Deaf', country: 'India', flag: '🇮🇳' },
                    { name: 'Fédération Nationale des Sourds', country: 'France', flag: '🇫🇷' },
                    { name: 'Deaf Australia', country: 'Australia', flag: '🇦🇺' }
                  ].map((org, i) => (
                    <div key={i} className="flex items-center gap-3 p-2 rounded-lg bg-muted/50">
                      <span className="text-xl">{org.flag}</span>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{org.name}</p>
                        <p className="text-xs text-muted-foreground">{org.country}</p>
                      </div>
                    </div>
                  ))}
                </div>
                <a
                  href="#"
                  className="flex items-center justify-center gap-2 mt-4 text-sm text-primary hover:underline"
                >
                  View all partners
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export default GlobalLanguagesSection;
