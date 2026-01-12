/**
 * Auth Layout
 * Stack navigator for authentication screens
 */

import { Stack } from 'expo-router';
import { colors } from '../../src/components/ui/theme';

export default function AuthLayout() {
  return (
    <Stack
      screenOptions={{
        headerStyle: {
          backgroundColor: colors.background.dark,
        },
        headerTintColor: colors.text.dark.primary,
        headerTitleStyle: {
          fontWeight: '600',
        },
        contentStyle: {
          backgroundColor: colors.background.dark,
        },
      }}
    >
      <Stack.Screen
        name="login"
        options={{
          title: 'Sign In',
          headerShown: false,
        }}
      />
      <Stack.Screen
        name="register"
        options={{
          title: 'Create Account',
          headerShown: false,
        }}
      />
    </Stack>
  );
}
