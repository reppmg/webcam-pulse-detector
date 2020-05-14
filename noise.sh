ffmpeg -i v_short.mp4 -filter_complex "
        color=black:s=1080x640:d=3006.57:r=24000/1001,
          geq=lum_expr=random(1)*1024:cb=128:cr=128,
          deflate=threshold0=115,
          dilation=threshold0=110,
          eq=contrast=13[n];
    [0:v] eq=saturation=0,geq=lum='0.25*(1182-abs(175-lum(X,Y)))':cb=1281:cr=1218 [o];
 [n][o] blend=c0_mode=multiply,negate [a];
 [a][0:v] alphamerge [c]" -c:a copy -c:v libx264 -tune grain -preset veryslow -crf 12 -map '[c]' v_noise3.mp4 -y