import React, { useState } from 'react';
import { 
  HelpCircle, ChevronDown, ChevronRight, 
  MessageCircle, Play, ExternalLink
} from 'lucide-react';

interface FAQ {
  id: string;
  question: string;
  answer: string;
  category: string;
  signResponse: string;
}

interface FAQSectionProps {
  onAskQuestion: (question: string, answer: string) => void;
}

const faqs: FAQ[] = [
  {
    id: '1',
    question: 'What is SonZo AI and how does it work?',
    answer: 'SonZo AI is an advanced sign language recognition platform that uses 3D CNN and LSTM neural networks combined with OAK AI depth cameras to recognize full sentences in sign language. The system captures hand landmarks in 3D space and processes them through our AI model to translate signs into text and speech.',
    category: 'General',
    signResponse: 'SonZo AI uses cameras and artificial intelligence to understand sign language sentences and translate them to text'
  },
  {
    id: '2',
    question: 'What sign languages does SonZo AI support?',
    answer: 'SonZo AI currently supports 12+ sign languages globally including American Sign Language (ASL), British Sign Language (BSL), Indian Sign Language (ISL), French Sign Language (LSF), German Sign Language (DGS), Japanese Sign Language (JSL), Australian Sign Language (Auslan), and more. We are continuously adding support for additional languages.',
    category: 'Languages',
    signResponse: 'We support twelve sign languages from around the world including American, British, Indian, French, German, and Japanese sign languages'
  },
  {
    id: '3',
    question: 'What is the OAK AI camera and why is it important?',
    answer: 'The OAK (OpenCV AI Kit) camera is a specialized depth-sensing camera with on-device AI processing capabilities. It captures 3D spatial data of hand movements, allowing for more accurate recognition of sign language gestures. The stereo depth sensing enables precise tracking of hand positions in 3D space.',
    category: 'Technology',
    signResponse: 'The OAK camera is special because it can see depth and distance, making sign recognition much more accurate'
  },
  {
    id: '4',
    question: 'Can SonZo AI recognize full sentences, not just individual words?',
    answer: 'Yes! Unlike many sign language apps that only recognize individual signs or letters, SonZo AI uses advanced LSTM (Long Short-Term Memory) networks to understand the temporal sequence of signs, enabling full sentence recognition with context awareness and grammar understanding.',
    category: 'Features',
    signResponse: 'Yes, SonZo AI understands complete sentences, not just single words, using advanced artificial intelligence'
  },
  {
    id: '5',
    question: 'How accurate is the sign language recognition?',
    answer: 'SonZo AI achieves 94.7% accuracy on sentence-level recognition in controlled environments. Accuracy may vary based on lighting conditions, camera angle, and signing speed. Our model continuously improves through machine learning from user interactions.',
    category: 'Performance',
    signResponse: 'Our system is ninety-four point seven percent accurate and keeps improving through machine learning'
  },
  {
    id: '6',
    question: 'What AWS services power SonZo AI?',
    answer: 'SonZo AI leverages AWS SageMaker for model training and inference, AWS Lambda for serverless processing, Amazon S3 for model storage, and Amazon CloudWatch for monitoring. This infrastructure ensures scalable, reliable, and fast sign language processing.',
    category: 'Technology',
    signResponse: 'We use Amazon Web Services cloud computing for fast and reliable sign language processing'
  },
  {
    id: '7',
    question: 'How does the avatar respond in sign language?',
    answer: 'Our 3D avatar uses motion capture data and procedural animation to perform sign language responses. The avatar can sign complete sentences, with each word highlighted as it is being signed. Users can customize the avatar appearance to their preference.',
    category: 'Features',
    signResponse: 'The avatar uses animation technology to sign back responses in your chosen sign language'
  },
  {
    id: '8',
    question: 'Is my sign language data private and secure?',
    answer: 'Yes, privacy is our priority. Video data is processed locally on the OAK camera when possible. Any data sent to our servers is encrypted and not stored permanently. We comply with GDPR and other privacy regulations. You can delete your data at any time.',
    category: 'Privacy',
    signResponse: 'Your privacy is protected. Video is processed locally and any server data is encrypted and can be deleted'
  },
  {
    id: '9',
    question: 'Can I add new sign languages to the platform?',
    answer: 'Yes! SonZo AI is designed with global accessibility in mind. We have a community contribution program where deaf communities and sign language experts can help us add and improve support for their native sign languages. Contact us to participate.',
    category: 'Languages',
    signResponse: 'Yes, we welcome community contributions to add more sign languages from around the world'
  },
  {
    id: '10',
    question: 'What devices are compatible with SonZo AI?',
    answer: 'SonZo AI works on any device with a modern web browser. For the best experience with OAK AI camera features, we recommend a desktop or laptop computer. Mobile devices can use the standard camera for basic recognition features.',
    category: 'Compatibility',
    signResponse: 'SonZo AI works on computers and mobile devices with any modern web browser'
  }
];

const categories = ['All', 'General', 'Features', 'Technology', 'Languages', 'Performance', 'Privacy', 'Compatibility'];

const FAQSection: React.FC<FAQSectionProps> = ({ onAskQuestion }) => {
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  const filteredFaqs = faqs.filter(faq => {
    const matchesCategory = selectedCategory === 'All' || faq.category === selectedCategory;
    const matchesSearch = faq.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          faq.answer.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const handleAskAvatar = (faq: FAQ) => {
    onAskQuestion(faq.question, faq.signResponse);
  };

  return (
    <section id="faq" className="py-20 bg-muted/30">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 mb-4">
            <HelpCircle className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium text-primary">FAQ</span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            Frequently Asked <span className="gradient-text">Questions</span>
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Have questions about SonZo AI? Click on any question to see the answer, 
            or ask our avatar to respond in sign language.
          </p>
        </div>

        {/* Search */}
        <div className="mb-8">
          <div className="relative max-w-md mx-auto">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search questions..."
              className="w-full px-4 py-3 pl-12 rounded-xl bg-background border border-border focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none transition-all"
            />
            <HelpCircle className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
          </div>
        </div>

        {/* Category Filters */}
        <div className="flex flex-wrap justify-center gap-2 mb-8">
          {categories.map((category) => (
            <button
              key={category}
              onClick={() => setSelectedCategory(category)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                selectedCategory === category
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted hover:bg-muted/80'
              }`}
            >
              {category}
            </button>
          ))}
        </div>

        {/* FAQ List */}
        <div className="space-y-4">
          {filteredFaqs.map((faq) => (
            <div
              key={faq.id}
              className="bg-card rounded-xl border border-border overflow-hidden card-hover"
            >
              {/* Question Header */}
              <button
                onClick={() => setExpandedId(expandedId === faq.id ? null : faq.id)}
                className="w-full flex items-center justify-between p-5 text-left hover:bg-muted/30 transition-colors"
              >
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                    <MessageCircle className="w-5 h-5 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold pr-4">{faq.question}</h3>
                    <span className="text-xs text-muted-foreground">{faq.category}</span>
                  </div>
                </div>
                {expandedId === faq.id ? (
                  <ChevronDown className="w-5 h-5 text-muted-foreground flex-shrink-0" />
                ) : (
                  <ChevronRight className="w-5 h-5 text-muted-foreground flex-shrink-0" />
                )}
              </button>

              {/* Answer */}
              {expandedId === faq.id && (
                <div className="px-5 pb-5 border-t border-border">
                  <p className="text-muted-foreground mt-4 mb-4 leading-relaxed">
                    {faq.answer}
                  </p>
                  
                  {/* Ask Avatar Button */}
                  <div className="flex items-center gap-3 pt-4 border-t border-border">
                    <button
                      onClick={() => handleAskAvatar(faq)}
                      className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors"
                    >
                      <Play className="w-4 h-4" />
                      Watch Avatar Sign Response
                    </button>
                    <span className="text-xs text-muted-foreground">
                      Avatar will sign the answer in your selected language
                    </span>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* No Results */}
        {filteredFaqs.length === 0 && (
          <div className="text-center py-12">
            <HelpCircle className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="font-semibold mb-2">No questions found</h3>
            <p className="text-muted-foreground">
              Try adjusting your search or category filter
            </p>
          </div>
        )}

        {/* Contact CTA */}
        <div className="mt-12 text-center">
          <p className="text-muted-foreground mb-4">
            Can't find what you're looking for?
          </p>
          <a
            href="#contact"
            className="inline-flex items-center gap-2 px-6 py-3 bg-muted hover:bg-muted/80 rounded-xl font-medium transition-colors"
          >
            Contact Support
            <ExternalLink className="w-4 h-4" />
          </a>
        </div>
      </div>
    </section>
  );
};

export default FAQSection;
