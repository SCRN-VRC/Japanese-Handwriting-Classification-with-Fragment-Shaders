Shader "HandwriteClassify/InputBuffer"
{
    Properties
    {
        _MainTex ("Input", 2D) = "black" {}
        _BufferTex ("Buffer", 2D) = "black" {}
        _ClearBtnTex ("Clear Button Input", 2D) = "black" {}
        _MaxDist ("Max Distance", Float) = 0.02
    }
    SubShader
    {
        Tags { "Queue"="Overlay+1" "ForceNoShadowCasting"="True" "IgnoreProjector"="True" }
        ZWrite Off
        ZTest Always
        Cull Front
        
        Pass
        {
            Lighting Off
            SeparateSpecular Off
            Fog { Mode Off }
            
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 5.0

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float3 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            Texture2D<float4> _BufferTex;
            float4 _BufferTex_TexelSize;

            Texture2D<float4> _MainTex;
            float4 _MainTex_TexelSize;

            Texture2D<float4> _ClearBtnTex;
            float4 _ClearBtnTex_TexelSize;

            float _MaxDist;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = float4(v.uv * 2 - 1, 0, 1);
                #ifdef UNITY_UV_STARTS_AT_TOP
                v.uv.y = 1-v.uv.y;
                #endif
                o.uv.xy = UnityStereoTransformScreenSpaceTex(v.uv);
                o.uv.z = (distance(_WorldSpaceCameraPos,
                    mul(unity_ObjectToWorld, float4(0,0,0,1)).xyz) > _MaxDist) ?
                    -1 : 1;
                o.uv.z = unity_OrthoParams.w ? o.uv.z : -1;
                return o;
            }

            float4 frag (v2f i) : SV_Target
            {
                clip(i.uv.z);
                // sample the texture
                float4 col = _MainTex.Load(int3(i.uv.xy * _MainTex_TexelSize.zw, 0));
                float4 buf = _BufferTex.Load(int3(i.uv.xy * _BufferTex_TexelSize.zw, 0));
                
                float3 touchPosCount = 0.0;
                for (uint i = 0; i < uint(_ClearBtnTex_TexelSize.z); i++)
                {
                    for (uint j = 0; j < uint(_ClearBtnTex_TexelSize.w); j++)
                    {
                        float hit = _ClearBtnTex[uint2(i, j)].r;
                        touchPosCount.xy += hit > 0.1 ? float2(i, j) : 0..xx;
                        touchPosCount.z += hit > 0.1 ? 1.0 : 0.0;
                    }
                }

                col = saturate(col * 2.0 + buf);
                col = touchPosCount.z > 5.0 ? 0..xxxx : col;

                return col;
            }
            ENDCG
        }
    }
}
