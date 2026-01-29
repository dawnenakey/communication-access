import { createClient } from '@supabase/supabase-js';


// Initialize database client
const supabaseUrl = 'https://brgskiwpwufizunvocie.databasepad.com';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjE3MDI4MjIzLThhYzktNDI1Mi1hOThkLTE0OTRmNjM4ODcxMCJ9.eyJwcm9qZWN0SWQiOiJicmdza2l3cHd1Zml6dW52b2NpZSIsInJvbGUiOiJhbm9uIiwiaWF0IjoxNzY5MzA3MDE1LCJleHAiOjIwODQ2NjcwMTUsImlzcyI6ImZhbW91cy5kYXRhYmFzZXBhZCIsImF1ZCI6ImZhbW91cy5jbGllbnRzIn0.EWrRxF4UQaTD-4RyYnjERCH_az9n6H75AA_6yBDmRY0';
const supabase = createClient(supabaseUrl, supabaseKey);


export { supabase };