import React, { useState, useCallback } from 'react';
import { 
  X, Search, Heart, Star, Download, 
  User, Users, Sparkles, Check, Loader2,
  Filter, Grid, List, Play, Video, Award
} from 'lucide-react';

interface Signer {
  id: string;
  name: string;
  category: 'Professional' | 'Educator' | 'Community' | 'Celebrity';
  imageUrl: string;
  description: string;
  languages: string[];
  specialties: string[];
  rating: number;
  signCount: number;
  isFavorite?: boolean;
  isPremium?: boolean;
  isVerified?: boolean;
}

interface AvatarMarketplaceProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectAvatar: (url: string) => void;
  currentAvatarUrl?: string;
}

// Realistic human signers gallery
const SIGNER_GALLERY: Signer[] = [
  {
    id: 'signer-1',
    name: 'Sarah Mitchell',
    category: 'Professional',
    imageUrl: 'https://d64gsuwffb70l.cloudfront.net/69757acc7df661eddaca039f_1769662049269_f23c9fd3.jpg',
    description: 'Certified ASL interpreter with 15 years of experience. Specializes in medical and legal interpretation.',
    languages: ['ASL', 'SEE'],
    specialties: ['Medical', 'Legal', 'Education'],
    rating: 4.9,
    signCount: 2500,
    isPremium: false,
    isVerified: true
  },
  {
    id: 'signer-2',
    name: 'Marcus Johnson',
    category: 'Professional',
    imageUrl: 'https://d64gsuwffb70l.cloudfront.net/69757acc7df661eddaca039f_1769662070071_b9b6621d.png',
    description: 'Native Deaf signer and ASL educator. Known for clear, expressive signing style.',
    languages: ['ASL', 'BSL'],
    specialties: ['Education', 'Entertainment', 'News'],
    rating: 4.8,
    signCount: 3200,
    isPremium: false,
    isVerified: true
  },
  {
    id: 'signer-3',
    name: 'Emily Chen',
    category: 'Educator',
    imageUrl: 'https://d64gsuwffb70l.cloudfront.net/69757acc7df661eddaca039f_1769662199341_f8b540c1.jpg',
    description: 'ASL professor and curriculum developer. Expert in teaching sign language to beginners.',
    languages: ['ASL', 'CSL', 'JSL'],
    specialties: ['Education', 'Children', 'Beginner'],
    rating: 4.9,
    signCount: 1800,
    isPremium: true,
    isVerified: true
  },
  {
    id: 'signer-4',
    name: 'David Williams',
    category: 'Professional',
    imageUrl: 'https://d64gsuwffb70l.cloudfront.net/69757acc7df661eddaca039f_1769662212613_20469d57.jpg',
    description: 'Broadcast interpreter for major news networks. Specializes in live interpretation.',
    languages: ['ASL'],
    specialties: ['News', 'Live Events', 'Politics'],
    rating: 4.7,
    signCount: 4100,
    isPremium: true,
    isVerified: true
  },
  {
    id: 'signer-5',
    name: 'Maria Rodriguez',
    category: 'Community',
    imageUrl: 'https://d64gsuwffb70l.cloudfront.net/69757acc7df661eddaca039f_1769662229973_66990e33.png',
    description: 'Bilingual interpreter (ASL/Spanish). Community advocate and accessibility consultant.',
    languages: ['ASL', 'LSM'],
    specialties: ['Community', 'Healthcare', 'Social Services'],
    rating: 4.8,
    signCount: 2100,
    isPremium: false,
    isVerified: true
  },
  {
    id: 'signer-6',
    name: 'James Thompson',
    category: 'Educator',
    imageUrl: 'https://d64gsuwffb70l.cloudfront.net/69757acc7df661eddaca039f_1769662070071_b9b6621d.png',
    description: 'Deaf Studies professor and researcher. Expert in ASL linguistics and Deaf culture.',
    languages: ['ASL', 'ISL'],
    specialties: ['Academic', 'Research', 'Linguistics'],
    rating: 4.6,
    signCount: 1500,
    isPremium: false,
    isVerified: true
  },
  {
    id: 'signer-7',
    name: 'Ashley Park',
    category: 'Celebrity',
    imageUrl: 'https://d64gsuwffb70l.cloudfront.net/69757acc7df661eddaca039f_1769662199341_f8b540c1.jpg',
    description: 'Deaf actress and ASL advocate. Featured in major streaming productions.',
    languages: ['ASL', 'KSL'],
    specialties: ['Entertainment', 'Acting', 'Storytelling'],
    rating: 4.9,
    signCount: 800,
    isPremium: true,
    isVerified: true
  },
  {
    id: 'signer-8',
    name: 'Robert Davis',
    category: 'Professional',
    imageUrl: 'https://d64gsuwffb70l.cloudfront.net/69757acc7df661eddaca039f_1769662212613_20469d57.jpg',
    description: 'VRI specialist and technology consultant. Pioneering remote interpretation services.',
    languages: ['ASL', 'Auslan'],
    specialties: ['Technology', 'Business', 'Remote'],
    rating: 4.7,
    signCount: 2800,
    isPremium: false,
    isVerified: true
  }
];

const categoryIcons = {
  Professional: Users,
  Educator: Award,
  Community: Heart,
  Celebrity: Sparkles
};

const AvatarMarketplace: React.FC<AvatarMarketplaceProps> = ({
  isOpen,
  onClose,
  onSelectAvatar,
  currentAvatarUrl
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [favorites, setFavorites] = useState<Set<string>>(new Set());
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [sortBy, setSortBy] = useState<'popular' | 'rating' | 'newest'>('popular');
  const [selectedSigner, setSelectedSigner] = useState<Signer | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Filter signers
  const filteredSigners = SIGNER_GALLERY.filter(signer => {
    const matchesSearch = searchQuery === '' || 
      signer.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      signer.specialties.some(s => s.toLowerCase().includes(searchQuery.toLowerCase())) ||
      signer.languages.some(l => l.toLowerCase().includes(searchQuery.toLowerCase()));
    const matchesCategory = !selectedCategory || signer.category === selectedCategory;
    return matchesSearch && matchesCategory;
  }).sort((a, b) => {
    if (sortBy === 'popular') return b.signCount - a.signCount;
    if (sortBy === 'rating') return b.rating - a.rating;
    return 0;
  });

  // Toggle favorite
  const toggleFavorite = useCallback((id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setFavorites(prev => {
      const newFavorites = new Set(prev);
      if (newFavorites.has(id)) {
        newFavorites.delete(id);
      } else {
        newFavorites.add(id);
      }
      return newFavorites;
    });
  }, []);

  // Select signer
  const handleSelectSigner = useCallback((signer: Signer) => {
    setIsLoading(true);
    setTimeout(() => {
      onSelectAvatar(signer.imageUrl);
      setIsLoading(false);
      onClose();
    }, 800);
  }, [onSelectAvatar, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-6xl max-h-[90vh] bg-card rounded-2xl shadow-2xl border border-border overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border bg-gradient-to-r from-violet-600/10 to-purple-600/10">
          <div>
            <h2 className="text-2xl font-bold flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                <Video className="w-5 h-5 text-white" />
              </div>
              Signer Marketplace
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
              Choose from {SIGNER_GALLERY.length} professional ASL signers for realistic video responses
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-muted transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Search & Filters */}
        <div className="p-4 border-b border-border bg-muted/30">
          <div className="flex flex-wrap gap-4 items-center">
            {/* Search */}
            <div className="relative flex-1 min-w-[200px]">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search signers, languages, specialties..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-background border border-border focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all"
              />
            </div>

            {/* Category Filters */}
            <div className="flex gap-2">
              {(['Professional', 'Educator', 'Community', 'Celebrity'] as const).map(category => {
                const Icon = categoryIcons[category];
                return (
                  <button
                    key={category}
                    onClick={() => setSelectedCategory(selectedCategory === category ? null : category)}
                    className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all ${
                      selectedCategory === category
                        ? 'bg-primary text-primary-foreground shadow-lg'
                        : 'bg-background border border-border hover:border-primary/50'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    {category}
                  </button>
                );
              })}
            </div>

            {/* Sort & View */}
            <div className="flex items-center gap-2">
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="px-3 py-2.5 rounded-xl bg-background border border-border text-sm"
              >
                <option value="popular">Most Signs</option>
                <option value="rating">Highest Rated</option>
                <option value="newest">Newest</option>
              </select>
              <div className="flex rounded-xl border border-border overflow-hidden">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`p-2.5 ${viewMode === 'grid' ? 'bg-primary text-primary-foreground' : 'bg-background hover:bg-muted'}`}
                >
                  <Grid className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`p-2.5 ${viewMode === 'list' ? 'bg-primary text-primary-foreground' : 'bg-background hover:bg-muted'}`}
                >
                  <List className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {/* Info Banner */}
          <div className="mb-6 p-4 rounded-xl bg-gradient-to-r from-emerald-500/10 to-teal-500/10 border border-emerald-500/20">
            <div className="flex items-start gap-3">
              <div className="w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center flex-shrink-0">
                <Video className="w-5 h-5 text-emerald-500" />
              </div>
              <div>
                <h3 className="font-semibold text-emerald-700 dark:text-emerald-400">GenASL Video Technology</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Our signers are recorded in high-definition video for the most realistic and accurate sign language representation. 
                  Each signer has thousands of pre-recorded sign clips that play seamlessly.
                </p>
              </div>
            </div>
          </div>

          {/* Signer Grid */}
          <div className={viewMode === 'grid' 
            ? 'grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4'
            : 'space-y-3'
          }>
            {filteredSigners.map(signer => (
              <div
                key={signer.id}
                onClick={() => setSelectedSigner(signer)}
                className={`group cursor-pointer rounded-xl border border-border overflow-hidden transition-all hover:border-primary hover:shadow-lg ${
                  viewMode === 'list' ? 'flex items-center p-4 gap-4' : ''
                } ${currentAvatarUrl === signer.imageUrl ? 'ring-2 ring-primary' : ''}`}
              >
                {/* Photo */}
                <div className={`relative ${viewMode === 'list' ? 'w-20 h-20 flex-shrink-0 rounded-lg overflow-hidden' : 'aspect-[3/4]'}`}>
                  <img 
                    src={signer.imageUrl} 
                    alt={signer.name}
                    className="w-full h-full object-cover"
                  />
                  {/* Overlay */}
                  <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-colors flex items-center justify-center opacity-0 group-hover:opacity-100">
                    <span className="px-3 py-1.5 rounded-full bg-white/90 text-black text-xs font-medium flex items-center gap-1">
                      <Play className="w-3 h-3" />
                      Preview
                    </span>
                  </div>
                  {/* Favorite Button */}
                  <button
                    onClick={(e) => toggleFavorite(signer.id, e)}
                    className={`absolute top-2 right-2 p-1.5 rounded-full transition-all ${
                      favorites.has(signer.id)
                        ? 'bg-red-500 text-white'
                        : 'bg-black/50 text-white/70 hover:text-white'
                    }`}
                  >
                    <Heart className={`w-4 h-4 ${favorites.has(signer.id) ? 'fill-current' : ''}`} />
                  </button>
                  {/* Premium Badge */}
                  {signer.isPremium && (
                    <div className="absolute top-2 left-2 px-2 py-0.5 rounded-full bg-gradient-to-r from-amber-500 to-orange-500 text-white text-xs font-medium">
                      Premium
                    </div>
                  )}
                  {/* Verified Badge */}
                  {signer.isVerified && !signer.isPremium && (
                    <div className="absolute top-2 left-2 px-2 py-0.5 rounded-full bg-blue-500 text-white text-xs font-medium flex items-center gap-1">
                      <Check className="w-3 h-3" />
                      Verified
                    </div>
                  )}
                  {/* Selected Indicator */}
                  {currentAvatarUrl === signer.imageUrl && (
                    <div className="absolute bottom-2 right-2 w-6 h-6 rounded-full bg-green-500 flex items-center justify-center">
                      <Check className="w-4 h-4 text-white" />
                    </div>
                  )}
                </div>

                {/* Info */}
                <div className={viewMode === 'list' ? 'flex-1' : 'p-3'}>
                  <h4 className="font-medium text-sm truncate">{signer.name}</h4>
                  <p className="text-xs text-muted-foreground mt-0.5">{signer.category}</p>
                  <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <Star className="w-3 h-3 text-amber-500 fill-amber-500" />
                      {signer.rating}
                    </span>
                    <span className="flex items-center gap-1">
                      <Video className="w-3 h-3" />
                      {signer.signCount.toLocaleString()} signs
                    </span>
                  </div>
                  {viewMode === 'grid' && (
                    <div className="flex flex-wrap gap-1 mt-2">
                      {signer.languages.slice(0, 2).map(lang => (
                        <span key={lang} className="px-1.5 py-0.5 rounded bg-primary/10 text-primary text-[10px] font-medium">
                          {lang}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {viewMode === 'list' && (
                  <>
                    <div className="flex flex-wrap gap-1">
                      {signer.languages.map(lang => (
                        <span key={lang} className="px-2 py-1 rounded-full bg-primary/10 text-primary text-xs font-medium">
                          {lang}
                        </span>
                      ))}
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleSelectSigner(signer);
                      }}
                      className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium"
                    >
                      Select
                    </button>
                  </>
                )}
              </div>
            ))}
          </div>

          {filteredSigners.length === 0 && (
            <div className="text-center py-12">
              <User className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">No signers found matching your criteria</p>
            </div>
          )}
        </div>

        {/* Signer Detail Modal */}
        {selectedSigner && (
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-10">
            <div className="w-full max-w-lg bg-card rounded-2xl shadow-2xl border border-border overflow-hidden">
              {/* Preview */}
              <div className="aspect-[4/3] relative">
                <img 
                  src={selectedSigner.imageUrl} 
                  alt={selectedSigner.name}
                  className="w-full h-full object-cover"
                />
                <button
                  onClick={() => setSelectedSigner(null)}
                  className="absolute top-4 right-4 p-2 rounded-lg bg-black/50 text-white hover:bg-black/70 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
                {selectedSigner.isVerified && (
                  <div className="absolute bottom-4 left-4 px-3 py-1.5 rounded-full bg-blue-500 text-white text-sm font-medium flex items-center gap-1.5">
                    <Check className="w-4 h-4" />
                    Verified Signer
                  </div>
                )}
              </div>

              {/* Details */}
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-xl font-bold">{selectedSigner.name}</h3>
                    <p className="text-sm text-muted-foreground">{selectedSigner.category} Signer</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="flex items-center gap-1 text-sm">
                      <Star className="w-4 h-4 text-amber-500 fill-amber-500" />
                      {selectedSigner.rating}
                    </span>
                    <span className="flex items-center gap-1 text-sm text-muted-foreground">
                      <Video className="w-4 h-4" />
                      {selectedSigner.signCount.toLocaleString()}
                    </span>
                  </div>
                </div>

                <p className="text-sm text-muted-foreground mb-4">
                  {selectedSigner.description}
                </p>

                <div className="mb-4">
                  <p className="text-xs font-medium text-muted-foreground mb-2">Languages</p>
                  <div className="flex flex-wrap gap-2">
                    {selectedSigner.languages.map(lang => (
                      <span key={lang} className="px-2 py-1 rounded-full bg-primary/10 text-primary text-xs font-medium">
                        {lang}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="mb-6">
                  <p className="text-xs font-medium text-muted-foreground mb-2">Specialties</p>
                  <div className="flex flex-wrap gap-2">
                    {selectedSigner.specialties.map(specialty => (
                      <span key={specialty} className="px-2 py-1 rounded-full bg-muted text-xs">
                        {specialty}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="flex gap-3">
                  <button
                    onClick={() => setSelectedSigner(null)}
                    className="flex-1 px-4 py-3 rounded-xl border border-border hover:bg-muted transition-colors font-medium"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => handleSelectSigner(selectedSigner)}
                    disabled={isLoading}
                    className="flex-1 px-4 py-3 rounded-xl bg-gradient-to-r from-violet-500 to-purple-600 text-white font-medium hover:opacity-90 transition-opacity disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Loading...
                      </>
                    ) : (
                      <>
                        <Check className="w-4 h-4" />
                        Select Signer
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AvatarMarketplace;
