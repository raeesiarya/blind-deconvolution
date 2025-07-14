// src/auth.ts
import { PublicClientApplication } from "@azure/msal-browser";

const msalInstance = new PublicClientApplication({
    auth: {
        clientId: "8c1aa411-72f2-4986-af18-96dc416e9075",  // Application (client) ID
        authority: "https://login.microsoftonline.com/160f751c-240c-405c-8e38-6150eff360d6",  // Tenant ID
        redirectUri: "http://localhost:5173/", // standard for Vite
    },
});

export async function loginWithMicrosoft(): Promise<string> {
    await msalInstance.initialize(); // 👈 Dette må med
    const result = await msalInstance.loginPopup({
        scopes: ["openid", "profile", "email"],
    });
    return result.idToken;
}