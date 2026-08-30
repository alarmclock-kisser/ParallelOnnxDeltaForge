$ErrorActionPreference = 'Stop'
$asmPath = "C:\Users\op\.nuget\packages\naudio\3.0.1\lib\net9.0\NAudio.dll"
$dir = Split-Path $asmPath
$assembly = [System.Reflection.Assembly]::LoadFrom($asmPath)
$mp = New-Object System.Reflection.MetadataLoadProvider -ArgumentList ([System.Reflection.MetadataAssemblyResolver]::CreateFromDirectoryPath($dir))
$asm = $mp.LoadFromAssemblyPath($asmPath)
$null = $assembly

function Show-Methods($type) {
	if (-not $type) { Write-Host "TYPE IS NULL"; return }
	Write-Host "=== $($type.FullName) ==="
	Write-Host "--- public methods ---"
	$type.GetMethods([System.Reflection.BindingFlags]::Public -bor [System.Reflection.BindingFlags]::Instance -bor [System.Reflection.BindingFlags]::DeclaredOnly) | ForEach-Object {
		$params = ($_.GetParameters() | ForEach-Object { $_.ParameterType.Name + ' ' + $_.Name }) -join ', '
		Write-Host ($_.Name + ' (' + $params + ')')
	}
	Write-Host "--- inherited interface methods ---"
	$type.GetInterfaces() | ForEach-Object { Write-Host "  iface: " + $_.FullName }
}

$isp = $asm.GetType("NAudio.Wave.SampleProviders.ISampleProvider")
Write-Host "ISampleProvider null? $($isp -eq $null)"
if ($isp) { Show-Methods $isp }

Write-Host ""
$af = $asm.GetType("NAudio.Wave.AudioFileReader")
Write-Host "AudioFileReader null? $($af -eq $null)"
if ($af) {
	Write-Host "--- AudioFileReader Read* methods ---"
	$af.GetMethods([System.Reflection.BindingFlags]::Public -bor [System.Reflection.BindingFlags]::Instance -bor [System.Reflection.BindingFlags]::DeclaredOnly) | Where-Object { $_.Name -match 'Read' } | ForEach-Object {
		$params = ($_.GetParameters() | ForEach-Object { $_.ParameterType.Name + ' ' + $_.Name }) -join ', '
		Write-Host ($_.Name + ' (' + $params + ')')
	}
}

Write-Host ""
$bwp = $asm.GetType("NAudio.Wave.BufferedWaveProvider")
Write-Host "BufferedWaveProvider null? $($bwp -eq $null)"
if ($bwp) {
	Write-Host "--- BufferedWaveProvider constructors ---"
	$bwp.GetConstructors() | ForEach-Object {
		$pfs = $_.GetParameters()
		$params = ''
		foreach ($p in $pfs) {
			$s = $p.ParameterType.Name + ' ' + $p.Name
			if ($p.HasDefaultValue) { $s = $s + ' = ' + $p.DefaultValue }
			$params = if ($params -eq '') { $s } else { $params + ', ' + $s }
		}
		Write-Host '(' + $params + ')'
	}
	Write-Host "--- BufferedWaveProvider properties ---"
	$bwp.GetProperties() | ForEach-Object { Write-Host $_.Name + ' : ' + $_.PropertyType.Name + ' get=' + $_.CanWrite }
}
