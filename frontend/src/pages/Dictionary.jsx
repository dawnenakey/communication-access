import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { toast } from "sonner";
import { useAuth, API } from "../App";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Textarea } from "../components/ui/textarea";
import { ScrollArea } from "../components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
  DialogClose,
} from "../components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "../components/ui/alert-dialog";
import { Avatar, AvatarFallback, AvatarImage } from "../components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../components/ui/dropdown-menu";
import { 
  Hand, Plus, Search, Pencil, Trash2, Upload, Image, 
  BookOpen, Camera, History, LogOut, User, Settings, Loader2,
  X, ChevronLeft
} from "lucide-react";

export default function Dictionary() {
  const { user } = useAuth();
  const navigate = useNavigate();
  
  const [signs, setSigns] = useState([]);
  const [filteredSigns, setFilteredSigns] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  
  // Add/Edit dialog states
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [selectedSign, setSelectedSign] = useState(null);
  const [formData, setFormData] = useState({ word: "", description: "" });
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isSaving, setIsSaving] = useState(false);
  
  const fileInputRef = useRef(null);

  useEffect(() => {
    loadSigns();
  }, []);

  useEffect(() => {
    if (searchQuery) {
      const filtered = signs.filter(sign => 
        sign.word.toLowerCase().includes(searchQuery.toLowerCase()) ||
        sign.description?.toLowerCase().includes(searchQuery.toLowerCase())
      );
      setFilteredSigns(filtered);
    } else {
      setFilteredSigns(signs);
    }
  }, [searchQuery, signs]);

  const loadSigns = async () => {
    setIsLoading(true);
    try {
      const response = await axios.get(`${API}/signs`);
      setSigns(response.data);
      setFilteredSigns(response.data);
    } catch (error) {
      toast.error("Failed to load dictionary");
    } finally {
      setIsLoading(false);
    }
  };

  const handleImageSelect = (event) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        toast.error("Please select an image file");
        return;
      }
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const resetForm = () => {
    setFormData({ word: "", description: "" });
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleAddSign = async () => {
    if (!formData.word.trim()) {
      toast.error("Please enter a word");
      return;
    }
    if (!selectedImage) {
      toast.error("Please select an image");
      return;
    }

    setIsSaving(true);
    try {
      const data = new FormData();
      data.append("word", formData.word.trim());
      data.append("description", formData.description.trim());
      data.append("image", selectedImage);

      await axios.post(`${API}/signs`, data, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      toast.success("Sign added successfully!");
      setIsAddDialogOpen(false);
      resetForm();
      loadSigns();
    } catch (error) {
      toast.error("Failed to add sign");
    } finally {
      setIsSaving(false);
    }
  };

  const handleEditSign = async () => {
    if (!formData.word.trim()) {
      toast.error("Please enter a word");
      return;
    }

    setIsSaving(true);
    try {
      const data = new FormData();
      data.append("word", formData.word.trim());
      data.append("description", formData.description.trim());
      if (selectedImage) {
        data.append("image", selectedImage);
      }

      await axios.put(`${API}/signs/${selectedSign.sign_id}`, data, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      toast.success("Sign updated successfully!");
      setIsEditDialogOpen(false);
      setSelectedSign(null);
      resetForm();
      loadSigns();
    } catch (error) {
      toast.error("Failed to update sign");
    } finally {
      setIsSaving(false);
    }
  };

  const handleDeleteSign = async (signId) => {
    try {
      await axios.delete(`${API}/signs/${signId}`);
      toast.success("Sign deleted successfully!");
      loadSigns();
    } catch (error) {
      toast.error("Failed to delete sign");
    }
  };

  const openEditDialog = (sign) => {
    setSelectedSign(sign);
    setFormData({ word: sign.word, description: sign.description || "" });
    setImagePreview(`data:${sign.image_type};base64,${sign.image_data}`);
    setIsEditDialogOpen(true);
  };

  const handleLogout = async () => {
    try {
      await axios.post(`${API}/auth/logout`);
      navigate("/");
    } catch (error) {
      navigate("/");
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 glass border-b border-white/5">
        <div className="flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-500 flex items-center justify-center">
              <Hand className="w-5 h-5 text-black" />
            </div>
            <span className="font-heading font-bold text-xl tracking-tight">SignSync AI</span>
          </div>

          <nav className="hidden md:flex items-center gap-2">
            <Button 
              variant="ghost" 
              onClick={() => navigate("/dashboard")}
              data-testid="nav-dashboard"
            >
              <Camera className="w-4 h-4 mr-2" />
              Translate
            </Button>
            <Button 
              variant="ghost" 
              onClick={() => navigate("/dictionary")}
              data-testid="nav-dictionary"
              className="text-foreground bg-white/5"
            >
              <BookOpen className="w-4 h-4 mr-2" />
              Dictionary
            </Button>
            <Button 
              variant="ghost" 
              onClick={() => navigate("/history")}
              data-testid="nav-history"
            >
              <History className="w-4 h-4 mr-2" />
              History
            </Button>
          </nav>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="gap-2" data-testid="user-menu-btn">
                <Avatar className="w-8 h-8">
                  <AvatarImage src={user?.picture} alt={user?.name} />
                  <AvatarFallback className="bg-primary/20 text-primary">
                    {user?.name?.charAt(0) || "U"}
                  </AvatarFallback>
                </Avatar>
                <span className="hidden md:inline font-medium">{user?.name}</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56 glass">
              <DropdownMenuItem className="cursor-pointer" disabled>
                <User className="w-4 h-4 mr-2" />
                Profile
              </DropdownMenuItem>
              <DropdownMenuItem className="cursor-pointer" disabled>
                <Settings className="w-4 h-4 mr-2" />
                Settings
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem 
                className="cursor-pointer text-destructive focus:text-destructive"
                onClick={handleLogout}
                data-testid="logout-btn"
              >
                <LogOut className="w-4 h-4 mr-2" />
                Logout
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      {/* Main Content */}
      <main className="p-6 lg:p-8">
        <div className="max-w-6xl mx-auto space-y-6">
          {/* Page Header */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={() => navigate("/dashboard")}
                className="mb-2 -ml-2"
              >
                <ChevronLeft className="w-4 h-4 mr-1" />
                Back to Dashboard
              </Button>
              <h1 className="font-heading font-bold text-3xl md:text-4xl tracking-tight">
                Sign Dictionary
              </h1>
              <p className="text-muted-foreground mt-1">
                Manage your custom ASL sign library
              </p>
            </div>
            
            <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
              <DialogTrigger asChild>
                <Button 
                  className="rounded-full bg-primary text-primary-foreground"
                  data-testid="add-sign-btn"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  Add New Sign
                </Button>
              </DialogTrigger>
              <DialogContent className="glass sm:max-w-md">
                <DialogHeader>
                  <DialogTitle className="font-heading">Add New Sign</DialogTitle>
                </DialogHeader>
                
                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Word</label>
                    <Input 
                      value={formData.word}
                      onChange={(e) => setFormData({ ...formData, word: e.target.value })}
                      placeholder="Enter the English word"
                      className="bg-black/20 border-white/10 rounded-xl"
                      data-testid="sign-word-input"
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Description (optional)</label>
                    <Textarea 
                      value={formData.description}
                      onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                      placeholder="Describe how to perform this sign"
                      className="bg-black/20 border-white/10 rounded-xl resize-none"
                      rows={3}
                      data-testid="sign-description-input"
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Sign Image</label>
                    <input 
                      type="file"
                      accept="image/*"
                      onChange={handleImageSelect}
                      ref={fileInputRef}
                      className="hidden"
                    />
                    
                    {imagePreview ? (
                      <div className="relative">
                        <img 
                          src={imagePreview}
                          alt="Preview"
                          className="w-full aspect-square object-cover rounded-xl"
                        />
                        <Button 
                          variant="ghost"
                          size="icon"
                          className="absolute top-2 right-2 bg-black/50 hover:bg-black/70 rounded-full"
                          onClick={() => {
                            setSelectedImage(null);
                            setImagePreview(null);
                            if (fileInputRef.current) fileInputRef.current.value = "";
                          }}
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      </div>
                    ) : (
                      <div 
                        onClick={() => fileInputRef.current?.click()}
                        className="border-2 border-dashed border-white/10 rounded-xl p-8 text-center cursor-pointer hover:border-cyan-400/30 hover:bg-cyan-400/5 transition-colors"
                        data-testid="image-upload-area"
                      >
                        <Image className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
                        <p className="text-sm text-muted-foreground">Click to upload image</p>
                      </div>
                    )}
                  </div>
                </div>
                
                <DialogFooter>
                  <DialogClose asChild>
                    <Button variant="ghost" className="rounded-full">Cancel</Button>
                  </DialogClose>
                  <Button 
                    onClick={handleAddSign}
                    disabled={isSaving}
                    className="rounded-full bg-primary text-primary-foreground"
                    data-testid="save-sign-btn"
                  >
                    {isSaving ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Saving...
                      </>
                    ) : (
                      "Add Sign"
                    )}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>

          {/* Search */}
          <div className="relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
            <Input 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search signs..."
              className="pl-12 h-12 bg-card/50 border-white/10 rounded-xl"
              data-testid="search-input"
            />
          </div>

          {/* Signs Grid */}
          {isLoading ? (
            <div className="flex items-center justify-center py-20">
              <Loader2 className="w-8 h-8 animate-spin text-cyan-400" />
            </div>
          ) : filteredSigns.length > 0 ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {filteredSigns.map((sign) => (
                <div 
                  key={sign.sign_id}
                  className="glass-card overflow-hidden group"
                  data-testid={`sign-card-${sign.sign_id}`}
                >
                  <div className="relative aspect-square">
                    <img 
                      src={`data:${sign.image_type};base64,${sign.image_data}`}
                      alt={sign.word}
                      className="w-full h-full object-cover"
                    />
                    
                    {/* Hover overlay */}
                    <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
                      <Button 
                        variant="ghost"
                        size="icon"
                        className="bg-white/10 hover:bg-white/20 rounded-full"
                        onClick={() => openEditDialog(sign)}
                        data-testid={`edit-sign-${sign.sign_id}`}
                      >
                        <Pencil className="w-4 h-4" />
                      </Button>
                      
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button 
                            variant="ghost"
                            size="icon"
                            className="bg-white/10 hover:bg-red-500/50 rounded-full"
                            data-testid={`delete-sign-${sign.sign_id}`}
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent className="glass">
                          <AlertDialogHeader>
                            <AlertDialogTitle>Delete Sign</AlertDialogTitle>
                            <AlertDialogDescription>
                              Are you sure you want to delete "{sign.word}"? This action cannot be undone.
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel className="rounded-full">Cancel</AlertDialogCancel>
                            <AlertDialogAction 
                              onClick={() => handleDeleteSign(sign.sign_id)}
                              className="rounded-full bg-destructive text-destructive-foreground"
                            >
                              Delete
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </div>
                  </div>
                  
                  <div className="p-3">
                    <h3 className="font-medium capitalize truncate">{sign.word}</h3>
                    {sign.description && (
                      <p className="text-xs text-muted-foreground truncate mt-1">
                        {sign.description}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="glass-card p-12 text-center">
              <BookOpen className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
              {searchQuery ? (
                <>
                  <h3 className="font-heading font-bold text-xl mb-2">No signs found</h3>
                  <p className="text-muted-foreground">
                    No signs match "{searchQuery}". Try a different search term.
                  </p>
                </>
              ) : (
                <>
                  <h3 className="font-heading font-bold text-xl mb-2">Your dictionary is empty</h3>
                  <p className="text-muted-foreground mb-6">
                    Start building your ASL sign library by adding new signs
                  </p>
                  <Button 
                    onClick={() => setIsAddDialogOpen(true)}
                    className="rounded-full bg-primary text-primary-foreground"
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    Add First Sign
                  </Button>
                </>
              )}
            </div>
          )}
        </div>
      </main>

      {/* Edit Dialog */}
      <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
        <DialogContent className="glass sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="font-heading">Edit Sign</DialogTitle>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Word</label>
              <Input 
                value={formData.word}
                onChange={(e) => setFormData({ ...formData, word: e.target.value })}
                placeholder="Enter the English word"
                className="bg-black/20 border-white/10 rounded-xl"
                data-testid="edit-word-input"
              />
            </div>
            
            <div className="space-y-2">
              <label className="text-sm font-medium">Description (optional)</label>
              <Textarea 
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                placeholder="Describe how to perform this sign"
                className="bg-black/20 border-white/10 rounded-xl resize-none"
                rows={3}
                data-testid="edit-description-input"
              />
            </div>
            
            <div className="space-y-2">
              <label className="text-sm font-medium">Sign Image</label>
              <input 
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                className="hidden"
                id="edit-image-input"
              />
              
              {imagePreview && (
                <div className="relative">
                  <img 
                    src={imagePreview}
                    alt="Preview"
                    className="w-full aspect-square object-cover rounded-xl"
                  />
                  <Button 
                    variant="ghost"
                    size="sm"
                    className="absolute bottom-2 right-2 bg-black/50 hover:bg-black/70 rounded-full"
                    onClick={() => document.getElementById('edit-image-input')?.click()}
                  >
                    <Upload className="w-4 h-4 mr-2" />
                    Change
                  </Button>
                </div>
              )}
            </div>
          </div>
          
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="ghost" className="rounded-full" onClick={() => {
                setSelectedSign(null);
                resetForm();
              }}>Cancel</Button>
            </DialogClose>
            <Button 
              onClick={handleEditSign}
              disabled={isSaving}
              className="rounded-full bg-primary text-primary-foreground"
              data-testid="update-sign-btn"
            >
              {isSaving ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                "Update Sign"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
