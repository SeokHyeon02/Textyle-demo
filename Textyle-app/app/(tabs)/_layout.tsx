import { Ionicons } from '@expo/vector-icons';
import { Tabs } from 'expo-router';

export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: '#8A2BE2',
        tabBarInactiveTintColor: 'gray',
        headerShown: false,
        headerTitleAlign: 'center',
      }}>
      
      <Tabs.Screen
        name="index"
        options={{
          title: '검색',
          tabBarIcon: ({ color, focused }) => 
            <Ionicons name={focused ? "search" : "search-outline"} size={26} color={color} />,
        }}
      />
      
      <Tabs.Screen
        name="bookmarks"
        options={{
          title: '찜',
          tabBarIcon: ({ color, focused }) => 
            <Ionicons name={focused ? "heart" : "heart-outline"} size={26} color={color} />,
        }}
      />
      
      <Tabs.Screen
        name="login"
        options={{
          title: '로그인',
          tabBarIcon: ({ color, focused }) => 
            <Ionicons name={focused ? "person" : "person-outline"} size={26} color={color} />,
        }}
      />
    </Tabs>
  );
}